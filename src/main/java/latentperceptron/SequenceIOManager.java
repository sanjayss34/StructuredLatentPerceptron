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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.Scanner;
import java.io.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.core.io.LineIO;
import edu.illinois.cs.cogcomp.sl.core.SLProblem;
import edu.illinois.cs.cogcomp.sl.util.FeatureVectorBuffer;
import edu.illinois.cs.cogcomp.sl.util.IFeatureVector;
public class SequenceIOManager {
	static Logger logger = LoggerFactory.getLogger(SequenceIOManager.class);

	/**
	 * reader for sequence problems 
	 * @author kchang10
	 */
	
	public static int numFeatures;
	public static int numLabels;
    public static int numLatent = 6;
    public static Map<String, Integer> labelIdMap;

	public static SLProblem readProblem(String fname, Boolean fixFeatureNum) throws IOException, Exception {
        try {
            Scanner scanner = new Scanner(new File("data/glove.6B/glove.6B.200d.txt"));
            labelIdMap = new HashMap<String, Integer>();
            String[] labelSet = {"I", "O", "B"};
            for (int i = 0; i < labelSet.length; i++) {
                labelIdMap.put(labelSet[i], i);
            }
            numLabels= labelSet.length;
            Map<String, float[]> vocab = new HashMap<String, float[]>();
            int vecLength = 0;
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] tokens = line.split(" ");
                if (vecLength == 0) {
                    vecLength = tokens.length-1;
                }
                float[] vec = new float[vecLength];
                for (int i = 1; i < tokens.length; i++) {
                    vec[i-1] = Float.parseFloat(tokens[i]);
                }
                vocab.put(tokens[0].toLowerCase(), vec);
            }
            scanner.close();
            scanner = new Scanner(new File("data/pos.txt"));
            Map<String, Integer> pos = new HashMap<String, Integer>();
            int posCount = 0;
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                pos.put(line.split("\n")[0], posCount);
                posCount++;
            }
            scanner.close();
            scanner = new Scanner(new File("data/"+fname));
            SLProblem sp = new SLProblem();
            List<IFeatureVector> currFvs = new ArrayList<IFeatureVector>();
            List<String> currLabels = new ArrayList<String>();
            String prevWord = null;
            String prevPos = null;
            /*String line = null;
            if (scanner.hasNextLine()) {
                line = scanner.nextLine();
            }*/
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                String[] tokens = line.split(" ");
                FeatureVectorBuffer fvb = new FeatureVectorBuffer();
                int currLength = 1;
                if (tokens.length > 1) {
                    float[] vec = new float[vecLength];
                    if (prevWord != null && vocab.containsKey(prevWord)) {
                        vec = vocab.get(prevWord);
                    }
                    for (int i = 0; i < vecLength; i++) {
                        fvb.addFeature(currLength+i, vec[i]);
                    }
                    currLength = vecLength;
                    vec = new float[vecLength];
                    String word = tokens[0].toLowerCase();
                    if (vocab.containsKey(word)) {
                        vec = vocab.get(word);
                    }
                    for (int i = 0; i < vecLength; i++) {
                        fvb.addFeature(currLength+i, vec[i]);
                    }
                    currLength += vecLength;
                    int posIndex = posCount;
                    if (prevPos != null && pos.containsKey(prevPos)) {
                        posIndex = pos.get(prevPos);
                    }
                    for (int i = 0; i < posCount+1; i++) {
                        if (i == posIndex) {
                            fvb.addFeature(currLength+i, 1);
                        }
                        else {
                            fvb.addFeature(currLength+i, 0);
                        }
                    }
                    currLength += posCount+1;
                    posIndex = posCount;
                    if (pos.containsKey(tokens[1])) {
                        posIndex = pos.get(tokens[1]);
                    }
                    for (int i = 0; i < posCount+1; i++) {
                        if (i == posIndex) {
                            fvb.addFeature(currLength+i, 1);
                        }
                        else {
                            fvb.addFeature(currLength+i, 0);
                        }
                    }
                    currLength += posCount+1;
                    /*if (!currFvbs.isEmpty()) {
                        for (int i = 0; i < vec.length; i++) {
                            currFvbs.get(currFvbs.size()-1).addFeature(currLength+i, vec[i]);
                        }
                        currLength += vec.length;
                        for (int i = 0; i < posCount+1; i++) {
                            if (i == posIndex) {
                                currFvbs.get(currFvbs.size()-1).addFeature(currLength+i, 1);
                            }
                            else {
                                currFvbs.get(currFvbs.size()-1).addFeature(currLength+i, 0);
                            }
                        }
                        currLength += posCount+1;
                    }*/
                    numFeatures = currLength;
                    prevWord = word;
                    prevPos = tokens[1];
                    currFvs.add(fvb.toFeatureVector());
                    // currFvbs.add(fvb);
                    currLabels.add(tokens[2]);
                }
                else {
                    /*if (!currFvbs.isEmpty()) {
                        for (int i = 0; i < vecLength; i++) {
                            currFvbs.get(currFvbs.size()-1).addFeature(currLength+i, 0);
                        }
                        currLength += vecLength;
                        for (int i = 0; i < posCount+1; i++) {
                            if (i == posCount) {
                                currFvbs.get(currFvbs.size()-1).addFeature(currLength+i, 1);
                            }
                            else {
                                currFvbs.get(currFvbs.size()-1).addFeature(currLength+i, 0);
                            }
                        }
                        currLength += posCount+1;
                    }*/
                    /*List<IFeatureVector> currFvs = new ArrayList<IFeatureVector>();
                    for (FeatureVectorBuffer v : currFvbs) {
                        currFvs.add(v.toFeatureVector());
                    }*/
                    SequenceInstance seq = new SequenceInstance(currFvs.toArray(new IFeatureVector[currFvs.size()]));
                    sp.instanceList.add(seq);
                    int[] labelArray = new int[currLabels.size()];
                    for (int i = 0; i < labelArray.length; i++) {
                        labelArray[i] = labelIdMap.get(currLabels.get(i));
                    }
                    sp.goldStructureList.add(new SequenceLabel(labelArray));
                    currLabels = new ArrayList<String>();
                    currFvs = new ArrayList<IFeatureVector>();
                    prevWord = null;
                    prevPos = null;
                }
            }
            scanner.close();
            return sp;
        }
        catch (IOException e) {
            System.out.println("Input error!");
            return null;
        }
    }
}
