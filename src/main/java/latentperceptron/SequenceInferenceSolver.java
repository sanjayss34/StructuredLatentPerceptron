/*******************************************************************************
 * University of Illinois/NCSA Open Source License
 * Copyright (c) 2010, 
 *
 * Developed by:
 * The Cognitive Computations Group
 * University of Illinois at Urbana-Champaign
 * http://cogcomp.cs.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
 * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution.
 * Neither the names of the Cognitive Computations Group, nor the University of Illinois at Urbana-Champaign, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
 *     
 *******************************************************************************/
package latentperceptron;

import edu.illinois.cs.cogcomp.sl.core.AbstractInferenceSolver;
import edu.illinois.cs.cogcomp.sl.core.IInstance;
import edu.illinois.cs.cogcomp.sl.core.IStructure;
import edu.illinois.cs.cogcomp.sl.util.WeightVector;

/**
 * A Viterbi inference solver
 * 
 * @author kchang10
 */
public class SequenceInferenceSolver extends
		AbstractInferenceSolver {

	private static final long serialVersionUID = 1L;	

	@Override
	public Object clone(){
		return new SequenceInferenceSolver();
	}
	
	@Override
	public IStructure getLossAugmentedBestStructure(
			WeightVector wv, IInstance input, IStructure gold)
			{
		SequenceLabel goldLabeledSeq = (SequenceLabel) gold;
		
		// initialization
		SequenceInstance seq = (SequenceInstance) input;
		
		int numOflabels = SequenceIOManager.numLabels;
        int numLatent = SequenceIOManager.numLatent;
		int numOfTokens = seq.baseFeatures.length;
		int numBaseFeatures = SequenceIOManager.numFeatures;
		
		float[][] dpTable = new float[2][numOflabels*numLatent];
		int[][] path = new int[numOfTokens][numOflabels*numLatent];
		
		int offset = (numBaseFeatures+1) * numOflabels * numLatent;
		
		// Viterbi algorithm
		for (int j = 0; j < numOflabels*numLatent; j++) {
			float priorScore = wv.get(numBaseFeatures * numOflabels*numLatent + j);
			float zeroOrderScore = wv.dotProduct(seq.baseFeatures[0], j*numBaseFeatures);
            if ((gold != null) && (j/numLatent != goldLabeledSeq.tags[0])) {
                zeroOrderScore = Float.NEGATIVE_INFINITY;
            }
                //+ ((gold !=null && j != goldLabeledSeq.tags[0])?1:0);
			dpTable[0][j] = priorScore + zeroOrderScore; 	 
			path[0][j] = -1;
		}
		
		for (int i = 1; i < numOfTokens; i++) {
			for (int j = 0; j < numOflabels*numLatent; j++) {
				float zeroOrderScore =  wv.dotProduct(seq.baseFeatures[i], j*numBaseFeatures);
                if ((gold != null) && (j/numLatent != goldLabeledSeq.tags[i])) {
                    zeroOrderScore = Float.NEGATIVE_INFINITY;
                }
                // + ((gold!=null && j != goldLabeledSeq.tags[i])?1:0);
				
				float bestScore = Float.NEGATIVE_INFINITY;
				for (int k = 0; k < numOflabels*numLatent; k++) {
					float candidateScore = dpTable[(i-1)%2][k] +  wv.get(offset + (k * numOflabels*numLatent + j));
					if (candidateScore > bestScore) {
						bestScore = candidateScore;
						path[i][j] = k;
					}
				}
				dpTable[i%2][j] = zeroOrderScore + bestScore;
			}
		}
		
		// find the best sequence of latent variables		
		int[] latentTags = new int[numOfTokens];
		
		int maxTag = 0;
		for (int i = 0; i < numOflabels*numLatent; i++)
			if (dpTable[(numOfTokens - 1)%2][i] > dpTable[(numOfTokens - 1)%2][maxTag]) 
				maxTag = i;
		
		latentTags[numOfTokens - 1] = maxTag;
		
		for (int i = numOfTokens - 1; i >= 1; i--) {
			latentTags[i-1] = path[i][latentTags[i]];
            // System.out.print(latentTags[i] + " ");
        }
        // System.out.println("");
        // Project latent variables onto label sequence
        /*int[] tags = new int[numOfTokens];
        for (int i = 0; i < numOfTokens; i++) {
            tags[i] = latentTags[i]/numLatent;
        }*/
		return new SequenceLabel(latentTags);
	}
		
	@Override
	public IStructure getBestStructure(WeightVector wv,
			IInstance input) {
		return getLossAugmentedBestStructure(wv, input, null);
	}
	@Override
	public float getLoss(IInstance ins, IStructure goldStructure,  IStructure structure){
		SequenceLabel goldLabeledSeq = (SequenceLabel) goldStructure;
		float loss = 0;
		for (int i = 0; i < goldLabeledSeq.tags.length; i++)
			if (((SequenceLabel) structure).tags[i] != goldLabeledSeq.tags[i])
				loss += 1.0f;
		return loss;
	}

    public SequenceLabel projectLatent(SequenceLabel h) {
        int[] tags = new int[h.tags.length];
        int numLatent = SequenceIOManager.numLatent;
        for (int i = 0; i < tags.length; i++) {
            tags[i] = h.tags[i]/numLatent;
        }
        return new SequenceLabel(tags);
    }
}
