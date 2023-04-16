// Thomas Lascaud
// 4/15/23
// A class that holds the information about cost funtion that is needed

import java.util.function.*;

public class CostFunction {
    // for a single one
    private BiFunction<Double, Double, Double> fn;
    // for a whole group
    private BiFunction<double[], double[], Double> groupFn;
    private BiFunction<Double, Double, Double> derivative;

    public static CostFunction MSE = new CostFunction(CostFunction::mseFn, CostFunction::mseGroupFn,
            CostFunction::mseDerivative);

    public CostFunction(BiFunction<Double, Double, Double> fn, BiFunction<double[], double[], Double> groupFn,
            BiFunction<Double, Double, Double> derivative) {
        this.fn = fn;
        this.groupFn = groupFn;
        this.derivative = derivative;
    }

    public double applyFn(double expected, double calc) {
        return fn.apply(calc, expected);
    }

    public double applyGroup(double[] expected, double[] calc) {
        return groupFn.apply(calc, expected);
    }

    public double applyDerivative(double calc, double expected) {
        return derivative.apply(calc, expected);
    }

    private static double mseFn(double calc, double expected) {
        // ! FiX
        return (calc - expected) * (calc - expected);
    }

    private static double mseGroupFn(double[] calc, double[] expected) {
        double total = 0;
        for (int i = 0; i < calc.length; i++) {
            // ! FIX
            total += (calc[i] - expected[i]) * (calc[i] - expected[i]);
        }
        return total / calc.length;
    }

    private static double mseDerivative(double calc, double expected) {
        return (calc - expected);
    }
}
