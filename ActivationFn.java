import java.util.function.Function;

public class ActivationFn {

    private Function<Double, Double> fn;
    private Function<Double, Double> derivative;

    public static ActivationFn SIGMOID = new ActivationFn(ActivationFn::sigmoid, ActivationFn::sigmoidDerivative);

    public ActivationFn(Function<Double, Double> fn, Function<Double, Double> derivative) {
        this.fn = fn;
        this.derivative = derivative;
    }

    public double apply(double num) {
        return fn.apply(num);
    }

    public double applyDerivative(double num) {
        return derivative.apply(num);
    }

    private static double sigmoid(double num) {
        return 1 / (1 + Math.exp(-1 * num));
    }

    private static double sigmoidDerivative(double num) {
        return sigmoid(num) * (1 - sigmoid(num));
    }

}
