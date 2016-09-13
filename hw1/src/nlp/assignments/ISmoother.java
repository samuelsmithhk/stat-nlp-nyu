package nlp.assignments;

import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * Created by samuelsmith on 11/9/2016.
 *
 * Implementations of this interface provide for various smoothing techniques on a Counter or CounterMap class
 */
public interface ISmoother {
    Counter<String> smoothCounter(Counter<String> toSmooth);
    CounterMap<String, String> smoothCounterMap(CounterMap<String, String> toSmooth);
}
