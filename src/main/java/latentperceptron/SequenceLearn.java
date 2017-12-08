package latentperceptron;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;
import edu.illinois.cs.cogcomp.sl.core.SLModel;
import edu.illinois.cs.cogcomp.sl.core.SLParameters;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;

public class SequenceLearn {
    public static int ITER = 40;
    public static int WV_SIZE = 10000;
    public static int K = 5;
    public static double alpha = 0.5;

    public static WeightVector avgWVs(ArrayList<WeightVector> wvs) {
        WeightVector wv = new WeightVector(WV_SIZE);
        for (int i = 0; i < wvs.size(); i++) {
            WeightVector wv_i = new WeightVector(WV_SIZE);
            wv_i.addDenseVector(wvs.get(i));
            wv_i.scale(i+1);
            wv.addDenseVector(wv_i);
        }
        wv.scale(1.0/(wvs.size()*(wvs.size()+1)/2));
        return wv;
    }

    public static WeightVector decayAvgWVs(ArrayList<WeightVector> wvs) {
        double mult = 1;
        WeightVector wv = new WeightVector(WV_SIZE);
        for (int i = 0; i < wvs.size(); i++) {
            WeightVector wvi = new WeightVector(WV_SIZE);
            wvi.addDenseVector(wvs.get(i));
            if (i < wvs.size()-1) {
                wvi.scale(alpha*mult);
            }
            else {
                wvi.scale((1-alpha)*mult);
            }
            wv.addDenseVector(wvi);
            mult *= (1-alpha);
        }
        return wv;
    }

    public static WeightVector weightedAvgWVs(ArrayList<WeightVector> wvs, ArrayList<Double> weights) {
        WeightVector wv = new WeightVector(WV_SIZE);
        double weightSum = 0;
        for (int i = 0; i < wvs.size(); i++) {
            WeightVector wvi = new WeightVector(WV_SIZE);
            wvi.addDenseVector(wvs.get(i));
            wvi.scale(weights.get(i));
            weightSum += weights.get(i);
            wv.addDenseVector(wvi);
        }
        wv.scale(weightSum);
        return wv;
    }

    public static double findAverage(List<Double> nums) {
        double avg = 0;
        for (double n : nums) {
            avg += n;
        }
        avg /= nums.size();
        return avg;
    }

    public static WeightVector train(SLProblem sp, SequenceFeatureGenerator fg, SequenceInferenceSolver infSolver) {
        List<IStructure> structures = sp.goldStructureList;
        List<IInstance> instances = sp.instanceList;
        WeightVector wv = new WeightVector(WV_SIZE);
        boolean[] holdOut = new boolean[instances.size()];
        Random randGen = new Random(System.currentTimeMillis());
        SLProblem holdOutSp = new SLProblem();
        for (int i = 0; i < instances.size(); i++) {
            if (randGen.nextDouble() < 0) {
                holdOut[i] = true;
                holdOutSp.instanceList.add(instances.get(i));
                holdOutSp.goldStructureList.add(structures.get(i));
            }
            else {
                holdOut[i] = false;
            }
        }
        ArrayList<Double> wvWeights = new ArrayList<Double>();
        ArrayList<WeightVector> all_wvs = new ArrayList<WeightVector>();
        for (int t = 0; t < ITER; t++) {
            for (int i = 0; i < instances.size(); i++) {
                if (holdOut[i]) {
                    continue;
                }
                SequenceInstance x = (SequenceInstance) instances.get(i);
                SequenceLabel y = (SequenceLabel) structures.get(i);
                SequenceLabel h = (SequenceLabel) infSolver.getBestStructure(wv, x);
                SequenceLabel y_hat = (SequenceLabel) infSolver.projectLatent(h);
                if (!y.equals(y_hat)) { // || infSolver.recentScore < 15000) {
                    SequenceLabel h_star = (SequenceLabel) infSolver.getLossAugmentedBestStructure(wv, x, y);
                    wv.addSparseFeatureVector(fg.getFeatureVector(x, h_star), 1);
                    wv.addSparseFeatureVector(fg.getFeatureVector(x, h), -1);
                }
            }
            all_wvs.add(wv);
            /*double acc = findAverage(wvWeights);
            try {
                acc = Evaluator.evaluate(holdOutSp, wv, infSolver, null);
                System.out.println(t + " " + acc);
            }
            catch (Exception e) {
                System.out.println(t);
            }
            wvWeights.add(acc);*/
            if (all_wvs.size() % K == 0) {
                // wv = weightedAvgWVs(all_wvs, wvWeights);
                wv = avgWVs(all_wvs);
                /*ArrayList<WeightVector> lastKWvs;
                if (all_wvs.size() < 2*K) {
                    lastKWvs = new ArrayList<WeightVector>(all_wvs);
                }
                else {
                    lastKWvs = new ArrayList<WeightVector>(all_wvs.subList(all_wvs.size()-2*K, all_wvs.size()));
                }
                wv = avgWVs(lastKWvs);*/

            }
        }

        return wv;
    }
}
