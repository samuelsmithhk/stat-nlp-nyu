package nlp.assignments;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * Created by samuelsmith on 11/9/2016.
 *
 * A language model that predicts the next word based on the previous n-1 words.
 */
public class NGramLanguageModel implements LanguageModel {

    private static final String start = "<S>", stop = "</S>", unknown = "*UNKNOWN*";

    private final ISmoother smoother;

    private final Counter<String> counter;
    private final LinkedHashMap<CounterMap<String, String>, Double> counterMaps;

    public NGramLanguageModel(int n, Collection<List<String>> trainingSentences, ISmoother smoother, double[] lambdas) {
        this.smoother = smoother;

        counter = new Counter<>();
        counterMaps = new LinkedHashMap<>();

        if (n > 1) createCounterMaps(n, lambdas);
        train(n, trainingSentences);
    }

    public NGramLanguageModel(int n, Collection<List<String>> trainingSentences, double[] lambdas) {
        this(n, trainingSentences, null, lambdas);
    }

    /**
     * Creates n countermaps, eg 0 for unigrams, 1 for bigrams, etc.
     * Lowest (index 0) is "most valuable" (highest lambda) for generating probability results
     * @param n
     */
    private void createCounterMaps(int n, double[] lambdas) {
        for (int i = 0; i < n - 1; i++) counterMaps.put(new CounterMap<>(), lambdas[i]);
    }

    private void train(int n, Collection<List<String>> trainingSentences) {

        for (List<String> sentence : trainingSentences) {

            List<String> stoppedSentence = new ArrayList<>(sentence);
            String[] previousWordBuffer = new String[n - 1];

            for (int i = 0; i < n - 1; i++) {
                stoppedSentence.add(0, start);
                previousWordBuffer[i] = start;
            }
            stoppedSentence.add(stop);

            for (int i = n - 1; i < stoppedSentence.size(); i++) {
                String word = stoppedSentence.get(i);
                counter.incrementCount(word, 1.0);

                int a = 0;
                for (CounterMap<String, String> counterMap : counterMaps.keySet()) {
                    StringBuilder sb = new StringBuilder();
                    for (int b = a; b < previousWordBuffer.length; b++) sb.append(previousWordBuffer[b]);
                    counterMap.incrementCount(sb.toString(), word, 1.0);
                    a += 1;
                }

                if (n > 1) {
                    System.arraycopy(previousWordBuffer, 1, previousWordBuffer, 0, previousWordBuffer.length - 1);
                    previousWordBuffer[previousWordBuffer.length - 1] = word;
                }
            }
        }

        counter.incrementCount(unknown, 1.0);

        if (smoother != null) {
            smoother.smoothCounter(counter);
            counterMaps.keySet().forEach(smoother::smoothCounterMap);
        }

        normalize();
    }

    private void normalize() {
        counter.normalize();
        for (CounterMap<String, String> counterMap : counterMaps.keySet())
            for (String previous : counterMap.keySet()) counterMap.getCounter(previous).normalize();
    }

    @Override
    public double getSentenceProbability(List<String> sentence) {
        double result = 1.0;
        int n = counterMaps.size();
        List<String> stoppedSentence = new ArrayList<>(sentence);
        String[] previousWordBuffer = new String[n];

        for (int i = 0; i < n; i++) {
            stoppedSentence.add(0, start);
            previousWordBuffer[i] = start;
        }
        stoppedSentence.add(stop);

        for (int i = n; i < stoppedSentence.size(); i++) {
            String word = stoppedSentence.get(i);
            result *= getNGramProbability(previousWordBuffer, word);

            if (n >= 1) {
                System.arraycopy(previousWordBuffer, 1, previousWordBuffer, 0, previousWordBuffer.length - 1);
                previousWordBuffer[previousWordBuffer.length - 1] = word;
            }
        }

        return result;
    }

    private double getNGramProbability(String[] previousWords, String word) {
        double result = 0.0, lambdaRemainder = 1.0;

        int a = 0;
        for (CounterMap<String, String> counterMap : counterMaps.keySet()) {
            StringBuilder sb = new StringBuilder();

            for (int b = a; b < previousWords.length; b++) sb.append(previousWords[b]);
            a += 1;

            double lambda = counterMaps.get(counterMap);
            result += lambda * counterMap.getCount(sb.toString(), word);
            lambdaRemainder -= lambda;
        }

        double unigramCount = counter.getCount(word);
        if (unigramCount == 0) unigramCount = counter.getCount(unknown); //unknown word

        return result + lambdaRemainder * unigramCount;
    }

    @Override
    public List<String> generateSentence() {
        return null;
    }
}
