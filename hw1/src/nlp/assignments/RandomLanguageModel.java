package nlp.assignments;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * Created by samuelsmith on 10/9/2016.
 */
public class RandomLanguageModel implements LanguageModel {

    private static final String stop = "</S>",
                                unknown = "*UNKNOWN*";

    private static final Random random = new Random();

    private final Counter<String> wordCounter;
    private final List<String> words;
    private final int wordSize;

    public RandomLanguageModel(Collection<List<String>> sentences) {
        wordCounter = new Counter<>();

        for (List<String> sentence : sentences) {
            List<String> stoppedSentence = new ArrayList<>(sentence);
            stoppedSentence.add(stop);

            for (String word : stoppedSentence) wordCounter.incrementCount(word, 1.0);
        }

        wordCounter.incrementCount(unknown, 1.0);
        wordCounter.normalize();

        words = new ArrayList<>(wordCounter.keySet());
        wordSize = words.size();
    }

    @Override
    public double getSentenceProbability(List<String> sentence) {
        return Math.random();
    }

    @Override
    public List<String> generateSentence() {
        List<String> sentence = new ArrayList<>();
        double likelyToStop = 0.1;
        boolean stopped = false;

        while (!stopped) {
            String word = generateWord();

            if (word.equals(stop)) {
                if (sentence.size() < 3) continue;
                stopped = true;
            }

            sentence.add(word);

            if (Math.random() <= likelyToStop) {
                if (sentence.size() < 3) continue;
                sentence.add(stop);
                stopped = true;
            }

            likelyToStop += 0.1;
        }

        return sentence;
    }

    private String generateWord() {
        return words.get(random.nextInt(wordSize));
    }
}
