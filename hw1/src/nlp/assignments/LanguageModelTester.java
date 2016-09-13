package nlp.assignments;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.text.NumberFormat;
import java.text.DecimalFormat;

import nlp.langmodel.LanguageModel;
import nlp.util.CommandLineUtils;

/**
 * This is the main harness for assignment 1. To run this harness, use
 * <p/>
 * java nlp.assignments.LanguageModelTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system. Second, find the point
 * in the main method (near the bottom) where an EmpiricalUnigramLanguageModel
 * is constructed. You will be writing new implementations of the LanguageModel
 * interface and constructing them there.
 */
public class LanguageModelTester {

	// HELPER CLASS FOR THE HARNESS, CAN IGNORE
	static class EditDistance {
		static double INSERT_COST = 1.0;
		static double DELETE_COST = 1.0;
		static double SUBSTITUTE_COST = 1.0;

		private double[][] initialize(double[][] d) {
			for (int i = 0; i < d.length; i++) {
				for (int j = 0; j < d[i].length; j++) {
					d[i][j] = Double.NaN;
				}
			}
			return d;
		}

		public double getDistance(List<? extends Object> firstList,
				List<? extends Object> secondList) {
			double[][] bestDistances = initialize(new double[firstList.size() + 1][secondList
					.size() + 1]);
			return getDistance(firstList, secondList, 0, 0, bestDistances);
		}

		private double getDistance(List<? extends Object> firstList,
				List<? extends Object> secondList, int firstPosition,
				int secondPosition, double[][] bestDistances) {
			if (firstPosition > firstList.size()
					|| secondPosition > secondList.size())
				return Double.POSITIVE_INFINITY;
			if (firstPosition == firstList.size()
					&& secondPosition == secondList.size())
				return 0.0;
			if (Double.isNaN(bestDistances[firstPosition][secondPosition])) {
				double distance = Double.POSITIVE_INFINITY;
				distance = Math.min(
						distance,
						INSERT_COST
								+ getDistance(firstList, secondList,
										firstPosition + 1, secondPosition,
										bestDistances));
				distance = Math.min(
						distance,
						DELETE_COST
								+ getDistance(firstList, secondList,
										firstPosition, secondPosition + 1,
										bestDistances));
				distance = Math.min(
						distance,
						SUBSTITUTE_COST
								+ getDistance(firstList, secondList,
										firstPosition + 1, secondPosition + 1,
										bestDistances));
				if (firstPosition < firstList.size()
						&& secondPosition < secondList.size()) {
					if (firstList.get(firstPosition).equals(
							secondList.get(secondPosition))) {
						distance = Math.min(
								distance,
								getDistance(firstList, secondList,
										firstPosition + 1, secondPosition + 1,
										bestDistances));
					}
				}
				bestDistances[firstPosition][secondPosition] = distance;
			}
			return bestDistances[firstPosition][secondPosition];
		}
	}

	// HELPER CLASS FOR THE HARNESS, CAN IGNORE
	static class SentenceCollection extends AbstractCollection<List<String>> {
		static class SentenceIterator implements Iterator<List<String>> {

			BufferedReader reader;

			public boolean hasNext() {
				try {
					return reader.ready();
				} catch (IOException e) {
					return false;
				}
			}

			public List<String> next() {
				try {
					String line = reader.readLine();
					String[] words = line.split("\\s+");
					List<String> sentence = new ArrayList<String>();
					for (int i = 0; i < words.length; i++) {
						String word = words[i];
						sentence.add(word.toLowerCase());
					}
					return sentence;
				} catch (IOException e) {
					throw new NoSuchElementException();
				}
			}

			public void remove() {
				throw new UnsupportedOperationException();
			}

			public SentenceIterator(BufferedReader reader) {
				this.reader = reader;
			}
		}

		String fileName;

		public Iterator<List<String>> iterator() {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(
						fileName));
				return new SentenceIterator(reader);
			} catch (FileNotFoundException e) {
				throw new RuntimeException("Problem with SentenceIterator for "
						+ fileName);
			}
		}

		public int size() {
			int size = 0;
			Iterator<List<String>> i = iterator();
			while (i.hasNext()) {
				size++;
				i.next();
			}
			return size;
		}

		public SentenceCollection(String fileName) {
			this.fileName = fileName;
		}

		public static class Reader {
			static Collection<List<String>> readSentenceCollection(
					String fileName) {
				return new SentenceCollection(fileName);
			}
		}

	}

	static double calculatePerplexity(LanguageModel languageModel,
			Collection<List<String>> sentenceCollection) {
		double logProbability = 0.0;
		double numSymbols = 0.0;
		for (List<String> sentence : sentenceCollection) {
			logProbability += Math.log(languageModel
					.getSentenceProbability(sentence)) / Math.log(2.0);
			numSymbols += sentence.size();
		}
		double avgLogProbability = logProbability / numSymbols;
		double perplexity = Math.pow(0.5, avgLogProbability);
		return perplexity;
	}

	static double calculateWordErrorRate(LanguageModel languageModel,
			List<SpeechNBestList> speechNBestLists, boolean verbose) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			List<String> bestGuess = null;
			double bestScore = Double.NEGATIVE_INFINITY;
			double numWithBestScores = 0.0;
			double distanceForBestScores = 0.0;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double score = Math.log(languageModel
						.getSentenceProbability(guess))
						+ (speechNBestList.getAcousticScore(guess) / 16.0);
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (score == bestScore) {
					numWithBestScores += 1.0;
					distanceForBestScores += distance;
				}
				if (score > bestScore || bestGuess == null) {
					bestScore = score;
					bestGuess = guess;
					distanceForBestScores = distance;
					numWithBestScores = 1.0;
				}
			}
			// double distance = editDistance.getDistance(correctSentence,
			// bestGuess);
			totalDistance += distanceForBestScores / numWithBestScores;
			totalWords += correctSentence.size();
			if (verbose) {
				System.out.println();
				displayHypothesis("GUESS:", bestGuess, speechNBestList,
						languageModel);
				displayHypothesis("GOLD:", correctSentence, speechNBestList,
						languageModel);
			}
		}
		return totalDistance / totalWords;
	}

	private static NumberFormat nf = new DecimalFormat("0.00E00");

	private static void displayHypothesis(String prefix, List<String> guess,
			SpeechNBestList speechNBestList, LanguageModel languageModel) {
		double acoustic = speechNBestList.getAcousticScore(guess) / 16.0;
		double language = Math.log(languageModel.getSentenceProbability(guess));
		System.out.println(prefix + "\tAM: " + nf.format(acoustic) + "\tLM: "
				+ nf.format(language) + "\tTotal: "
				+ nf.format(acoustic + language) + "\t" + guess);
	}

	static double calculateWordErrorRateLowerBound(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double bestDistance = Double.POSITIVE_INFINITY;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (distance < bestDistance)
					bestDistance = distance;
			}
			totalDistance += bestDistance;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateUpperBound(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double worstDistance = Double.NEGATIVE_INFINITY;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (distance > worstDistance)
					worstDistance = distance;
			}
			totalDistance += worstDistance;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateRandomChoice(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double sumDistance = 0.0;
			double numGuesses = 0.0;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				sumDistance += distance;
				numGuesses += 1.0;
			}
			totalDistance += sumDistance / numGuesses;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static Collection<List<String>> extractCorrectSentenceList(
			List<SpeechNBestList> speechNBestLists) {
		Collection<List<String>> correctSentences = new ArrayList<List<String>>();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			correctSentences.add(speechNBestList.getCorrectSentence());
		}
		return correctSentences;
	}

	static Set<String> extractVocabulary(
			Collection<List<String>> sentenceCollection) {
		Set<String> vocabulary = new HashSet<String>();
		for (List<String> sentence : sentenceCollection) {
			for (String word : sentence) {
				vocabulary.add(word);
			}
		}
		return vocabulary;
	}

	public static void mainOld(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Set up default parameters and settings
		String basePath = ".";
		String model = "baseline";
		boolean verbose = false;

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// A string descriptor of the model to use
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
		}
		System.out.println("Using model: " + model);

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}
		if (argMap.containsKey("-quiet")) {
			verbose = false;
		}

		// Read in all the assignment data
		String trainingSentencesFile = "/treebank-sentences-spoken-train.txt";
		String speechNBestListsPath = "/wsj_n_bst";
		Collection<List<String>> trainingSentenceCollection = SentenceCollection.Reader
				.readSentenceCollection(basePath + trainingSentencesFile);
		Set<String> trainingVocabulary = extractVocabulary(trainingSentenceCollection);
		List<SpeechNBestList> speechNBestLists = SpeechNBestList.Reader
				.readSpeechNBestLists(basePath + speechNBestListsPath,
						trainingVocabulary);

		// String validationSentencesFile =
		// "/treebank-sentences-spoken-validate.txt";
		// Collection<List<String>> validationSentenceCollection =
		// SentenceCollection.Reader.readSentenceCollection(basePath +
		// validationSentencesFile);

		 String testSentencesFile = "/treebank-sentences-spoken-test.txt";
		 Collection<List<String>> testSentenceCollection =
		 SentenceCollection.Reader.readSentenceCollection(basePath +
		 testSentencesFile);

		// Build the language model
		LanguageModel languageModel = null;
		if (model.equalsIgnoreCase("baseline")) {
			languageModel = new EmpiricalUnigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("sri")) {
			languageModel = new SriLanguageModel(argMap.get("-sri"));
		} else if (model.equalsIgnoreCase("bigram")) {
			languageModel = new EmpiricalBigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("trigram")) {
			languageModel = new EmpiricalTrigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("katz-bigram")) {
			languageModel = new KatzBigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("katz-trigram")) {
			languageModel = new KatzTrigramLanguageModel(
					trainingSentenceCollection);
		} else {
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		// Evaluate the language model
		 double wsjPerplexity = calculatePerplexity(languageModel, testSentenceCollection);
		double hubPerplexity = calculatePerplexity(languageModel,
				extractCorrectSentenceList(speechNBestLists));
		 System.out.println("WSJ Perplexity:  " + wsjPerplexity);
		System.out.println("HUB Perplexity:  " + hubPerplexity);
		System.out.println("WER Baselines:");
		System.out.println("  Best Path:  "
				+ calculateWordErrorRateLowerBound(speechNBestLists));
		System.out.println("  Worst Path: "
				+ calculateWordErrorRateUpperBound(speechNBestLists));
		System.out.println("  Avg Path:   "
				+ calculateWordErrorRateRandomChoice(speechNBestLists));
		double wordErrorRate = calculateWordErrorRate(languageModel,
				speechNBestLists, verbose);
		System.out.println("HUB Word Error Rate: " + wordErrorRate);
		System.out.println("Generated Sentences:");
			 System.out.println("  " + languageModel.generateSentence());
	}

	public static void main(String[] args) throws IOException {
		String	basePath = CommandLineUtils.simpleCommandLineParser(args).get("-path"),
				trainingFile = "/treebank-sentences-spoken-train.txt",
				validationFile = "/treebank-sentences-spoken-validate.txt",
				testFile = "/treebank-sentences-spoken-test.txt",
				wsjPath =  "/wsj_n_bst";

		Collection<List<String>> 	trainingSentences = SentenceCollection.Reader.readSentenceCollection(basePath
											+ trainingFile),
									validationSentences = SentenceCollection.Reader.readSentenceCollection(basePath
											+ validationFile),
									testSentences = SentenceCollection.Reader.readSentenceCollection(basePath
											+ testFile);

		Set<String> trainingVocabulary = extractVocabulary(trainingSentences);
		List<SpeechNBestList> speechNBestLists = SpeechNBestList.Reader.readSpeechNBestLists(basePath + wsjPath,
				trainingVocabulary);

		//build the language models
		System.out.println("LOADING MODELS");
		Map<String, LanguageModel> models = new LinkedHashMap<>();

		LaplaceSmoother laplace = new LaplaceSmoother();

		models.put("unigram", new NGramLanguageModel(1, trainingSentences, null));
		System.out.println("UNIGRAM LOADED");
		models.put("unigram-laplace", new NGramLanguageModel(1, trainingSentences, laplace ,null));
		System.out.println("UNIGRAM-LAPLACE LOADED");


		models.put("bigram", new NGramLanguageModel(2, trainingSentences, new double[]{0.7}));
		System.out.println("BIGRAM LOADED");
		models.put("bigram-laplace", new NGramLanguageModel(2, trainingSentences, laplace, new double[]{0.6}));
		System.out.println("BIGRAM-LAPLACE LOADED");

		models.put("trigram", new NGramLanguageModel(3, trainingSentences, new double[]{0.5, 0.3}));
		System.out.println("TRIGRAM LOADED");
		models.put("trigram-laplace", new NGramLanguageModel(3, trainingSentences, laplace, new double[]{0.5, 0.3}));
		System.out.println("TRIGRAM-LAPLACE LOADED");

		models.put("quadgram", new NGramLanguageModel(4, trainingSentences, new double[]{0.28, 0.27, 0.17}));
		System.out.println("QUADGRAM LOADED");
		models.put("quadgram-laplace", new NGramLanguageModel(4, trainingSentences, laplace, new double[]{0.26, 0.26, 0.18}));
		System.out.println("QUADGRAM-LAPLACE LOADED");

		models.put("quintgram", new NGramLanguageModel(5, trainingSentences, new double[]{0.3, 0.2, 0.15, 0.1}));
		System.out.println("QUINTGRAM LOADED");
		models.put("quintgram-laplace", new NGramLanguageModel(5, trainingSentences, laplace, new double[]{0.3, 0.2, 0.15, 0.1}));
		System.out.println("QUINTGRAM-LAPLACE LOADED");

		//calculate and display their hub perplexity scores, and also track time taken;
		System.out.println("\n\n--PERPLEXITY");
		double 	uniPerp = calculatePerplexity(models.get("unigram"), testSentences),
				uniLPerp = calculatePerplexity(models.get("unigram-laplace"), testSentences),

				biPerp = calculatePerplexity(models.get("bigram"), testSentences),
				biLPerp = calculatePerplexity(models.get("bigram-laplace"), testSentences),

				triPerp = calculatePerplexity(models.get("trigram"), testSentences),
				triLPerp = calculatePerplexity(models.get("trigram-laplace"), testSentences),

				quadPerp = calculatePerplexity(models.get("quadgram"), testSentences),
				quadLPerp = calculatePerplexity(models.get("quadgram-laplace"), testSentences),

				quintPerp = calculatePerplexity(models.get("quintgram"), testSentences),
				quintLPerp = calculatePerplexity(models.get("quintgram-laplace"), testSentences);

		String perpOutMessage =
								"----UNIGRAM\n" +
										"--------NO SMOOTHING:         [" + uniPerp + "]\n" +
										"--------LAPLACE SMOOTHING:    [" + uniLPerp + "]\n" +
								"----BIGRAM\n" +
										"--------NO SMOOTHING:         [" + biPerp + "]\n" +
										"--------LAPLACE SMOOTHING:    [" + biLPerp + "]\n" +
								"----TRIGRAM\n" +
										"--------NO SMOOTHING:         [" + triPerp + "]\n" +
										"--------LAPLACE SMOOTHING:    [" + triLPerp + "]\n" +
								"----QUADGRAM\n" +
										"--------NO SMOOTHING:         [" + quadPerp + "]\n" +
										"--------LAPLACE SMOOTHING:    [" + quadLPerp + "]\n" +
								"----QUINTGRAM\n" +
										"--------NO SMOOTHING:         [" + quintPerp + "]\n" +
										"--------LAPLACE SMOOTHING:    [" + quintLPerp + "]\n";

		System.out.println(perpOutMessage);

		System.out.println("\n--WORD ERROR RATE");
		double 	uniWER = calculateWordErrorRate(models.get("unigram"), speechNBestLists, false),
				uniLWER = calculateWordErrorRate(models.get("unigram-laplace"), speechNBestLists, false),

		 		biWER =  calculateWordErrorRate(models.get("bigram"), speechNBestLists, false),
				biLWER = calculateWordErrorRate(models.get("bigram-laplace"), speechNBestLists, false),

				triWER = calculateWordErrorRate(models.get("trigram"), speechNBestLists, true),
				triLWER = calculateWordErrorRate(models.get("trigram-laplace"), speechNBestLists, false),

				quadWER = calculateWordErrorRate(models.get("quadgram"), speechNBestLists, false),
				quadLWER = calculateWordErrorRate(models.get("quadgram-laplace"), speechNBestLists, false),

				quintWER = calculateWordErrorRate(models.get("quintgram"), speechNBestLists, false),
				quintLWER = calculateWordErrorRate(models.get("quintgram-laplace"), speechNBestLists, false);

		String werOutMessage =
				"----UNIGRAM\n" +
						"--------NO SMOOTHING:         [" + uniWER + "]\n" +
						"--------LAPLACE SMOOTHING:    [" + uniLWER + "]\n" +
						"----BIGRAM\n" +
						"--------NO SMOOTHING:         [" + biWER + "]\n" +
						"--------LAPLACE SMOOTHING:    [" + biLWER + "]\n" +
						"----TRIGRAM\n" +
						"--------NO SMOOTHING:         [" + triWER + "]\n" +
						"--------LAPLACE SMOOTHING:    [" + triLWER + "]\n" +
						"----QUADGRAM\n" +
						"--------NO SMOOTHING:         [" + quadWER + "]\n" +
						"--------LAPLACE SMOOTHING:    [" + quadLWER + "]\n" +
						"----QUINTGRAM\n" +
						"--------NO SMOOTHING:         [" + quintWER + "]\n" +
						"--------LAPLACE SMOOTHING:    [" + quintLWER + "]\n";

		System.out.println(werOutMessage);
	}
}
