package latentperceptron;

import java.util.List;
import java.util.ArrayList;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;

public class SequenceLearn {
    public static int ITER = 20;
    public static int WV_SIZE = 10000;
    public static int K = ITER/2;

    public static WeightVector avgWVs(ArrayList<WeightVector> wvs) {
        WeightVector wv = new WeightVector(WV_SIZE);
        for (WeightVector wv_i : wvs) {
            wv.addDenseVector(wv_i);
        }
        wv.scale(1.0/wvs.size());
        return wv;
    }

    public static WeightVector train(SLProblem sp, SequenceFeatureGenerator fg, SequenceInferenceSolver infSolver) {
        List<IStructure> structures = sp.goldStructureList;
        List<IInstance> instances = sp.instanceList;
        WeightVector wv = new WeightVector(WV_SIZE);
        List<WeightVector> all_wvs = new ArrayList<WeightVector>();
        for (int t = 0; t < ITER; t++) {
            System.out.println(t);
            for (int i = 0; i < instances.size(); i++) {
                SequenceInstance x = (SequenceInstance) instances.get(i);
                SequenceLabel y = (SequenceLabel) structures.get(i);
                SequenceLabel h = (SequenceLabel) infSolver.getBestStructure(wv, x);
                SequenceLabel y_hat = (SequenceLabel) infSolver.projectLatent(h);
                if (!y.equals(y_hat)) {
                    SequenceLabel h_star = (SequenceLabel) infSolver.getLossAugmentedBestStructure(wv, x, y);
                    wv.addSparseFeatureVector(fg.getFeatureVector(x, h_star), 1);
                    wv.addSparseFeatureVector(fg.getFeatureVector(x, h), -1);
                }
            }
            all_wvs.add(wv);
            if (all_wvs.size() % K == 0) {
                wv = avgWVs((ArrayList) all_wvs);
            }
        }

        return wv;
    }
}
