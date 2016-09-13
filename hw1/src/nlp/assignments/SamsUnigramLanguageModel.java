package nlp.assignments;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Created by samuelsmith on 10/9/2016.
 */
public class SamsUnigramLanguageModel implements LanguageModel {

    private static final String stop = "</S>",
                                unknown = "*UNKNOWN*";

    private final Counter<String> counter;

    public SamsUnigramLanguageModel(Collection<List<String>> sentences) {
        counter = new Counter<>();

        for (List<String> sentence : sentences) {
            for (String word : sentence) counter.incrementCount(word, 1.0);
            counter.incrementCount(stop, 1.0);
        }

        counter.incrementCount(unknown, 1.0);
        counter.normalize();
    }

    private double getWordProbability(String word) {
        double count = counter.getCount(word);
        if (count == 0) return counter.getCount(unknown);
        return count;
    }


    @Override
    public double getSentenceProbability(List<String> sentence) {
        List<String> stoppedSentence = new ArrayList<>(sentence);
        stoppedSentence.add(stop);
        double probability = 1.0;
        for (String word : stoppedSentence) {
            probability *= getWordProbability(word);
        }

        return probability;
    }

    private String generateWord() {
        double sample = Math.random();
        double sum = 0.0;
        for (String word : counter.keySet()) {
            sum += counter.getCount(word);
            if (sum > sample) {
                return word;
            }
        }
        return unknown;
    }

    public List<String> generateSentence() {
        List<String> sentence = new ArrayList<String>();
        String word = generateWord();
        while (!word.equals(stop)) {
            sentence.add(word);
            word = generateWord();
        }
        return sentence;
    }
}
