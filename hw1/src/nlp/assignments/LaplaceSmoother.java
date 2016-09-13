package nlp.assignments;

import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * Created by samuelsmith on 12/9/2016.
 */
public class LaplaceSmoother implements ISmoother{

    @Override
    public Counter<String> smoothCounter(Counter<String> toSmooth) {
        toSmooth.incrementAll(toSmooth.keySet(), 1.0);
        return toSmooth;
    }

    @Override
    public CounterMap<String, String> smoothCounterMap(CounterMap<String, String> toSmooth) {
        toSmooth.incrementAll(1.0);
        return toSmooth;
    }
}
