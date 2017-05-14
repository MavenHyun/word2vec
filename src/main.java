import java.util.ArrayList;
import java.util.Vector;

import Jama.Matrix;
import Jama.util.Maths;


/**
 * Created by Maven Hyun on 2017-05-02.
 */
public class main
{
    public static void main(String[] args) throws Exception
    {
        word2vec maven = new word2vec(100, 4, 0.001);
        maven.tokenize("text98.txt"); /*any text*/
        maven.train(70);
        return;
    }
    /**let's say there are 10 words**/





}
