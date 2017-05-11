/**
 * Created by Maven Hyun on 2017-05-02.
 */
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;


import java.util.concurrent.ExecutionException;

import Jama.Matrix;
import Jama.util.Maths;

import org.python.util.PythonInterpreter;
import org.python.core.*;



import com.aliasi.tokenizer.LowerCaseTokenizerFactory;
import com.aliasi.tokenizer.PorterStemmerTokenizerFactory;
import com.aliasi.tokenizer.Tokenizer;
import com.aliasi.tokenizer.TokenizerFactory;
import org.apache.lucene.analysis.core.KeywordTokenizer;

import org.apache.lucene.analysis.*;
import org.tartarus.snowball.ext.PorterStemmer;


public class word2vec
{


    public int N = 0;
    public int V = 0;
    public int C = 0;
    public int dict_capacity = 20000; /*10000*/
    public double learning_rate = 0.88;

    public Map<String, double[]> dict = new HashMap<String, double[]>();
    public ArrayList<String> context_words = new ArrayList<String>();
    public ArrayList<String> training_set = new ArrayList<String>();
    public Set<String> stopwords = new HashSet<String>();

    public Matrix weight = Matrix.random(V, N);
    public Matrix weight2 = Matrix.random(N, V);

    public Matrix input = new Matrix(V,1);
    public Matrix hidden = new Matrix(N,1);
    public Matrix output = new Matrix(V,1);
    public Matrix error = new Matrix(V,1);

    public void tokenize(String file)
    {
        try
        {
            Document parser;
            FileReader f = new FileReader(file);
            BufferedReader b = new BufferedReader(f);
            String line = null;
            while ((line = b.readLine()) != null)
            {
                try
                {
                    String text = "";
                    parser = Jsoup.connect(line).get();
                    Elements elements = parser.select(".zn-body__paragraph");
                    Element heading = elements.get(1);
                    text += heading.text();
                    for (int i = 2; i < elements.size(); i++ )
                    {
                        text += elements.get(i).text();
                    }
                    StringTokenizer tokens = new StringTokenizer(text.toLowerCase());
                    Stemmer porter = new Stemmer();
                    for (int i = 0; i < tokens.countTokens(); i++)
                    {
                        String token = tokens.nextToken();
                        if (!stopwords.contains(token))
                        {
                            porter.add(token);
                            String result = porter.stem();
                            if (!training_set.contains(result)) training_set.add(result);
                            porter.clear();
                        }
                    }

                    System.out.print("Parsing Complete!\n");
                }
                catch(Exception e) {}
            }
            V = training_set.size();
            weight = Matrix.random(V, N);
            weight2 = Matrix.random(N, V);
            copy_to_dict();
        }
        catch(Exception e) {}
    }

    word2vec(int feature, int context)
    {
        N = feature;
        C = context;
        try
        {
            FileReader f = new FileReader("stopword.txt");
            BufferedReader b = new BufferedReader(f);
            String line = null;
            while ((line = b.readLine()) != null)
            {
                stopwords.add(line);
            }
        }
        catch(Exception e) {}
    }

    public void get_C(ArrayList<String> word_list)
    {
        C = word_list.size();
    }

    void copy_to_dict()
    {
        for (String word : training_set)
        {
            double array[] = new double[V];
            array[dict.size()] = 1;
            dict.put(word, array);
        }
    }

    public Matrix vectorize(String word)
    {
        double[] array = dict.get(word);
        Matrix vector = new Matrix(array, 1);
        return vector.transpose();
    }

    public Matrix input_layer()
    {
        Matrix vector = new Matrix(V, 1);
        for (String word : context_words)
        {
            vector.plusEquals(vectorize(word));
        }
        input = vector;
        return vector;
    }

    public Matrix hidden_layer()
    {
        Matrix vector = new Matrix(N, 1);
        vector = (weight.transpose()).times(input_layer());
        vector = vector.times(1/(double)C);
        hidden = vector;
        return vector;
    }

    public Matrix output_layer(ArrayList<String> word_list)
    {
        Matrix vector = new Matrix(V, 1);
        vector = (weight2.transpose()).times(hidden_layer());
        double sum = 0;
        for (int i = 0; i < V; i++)
        {
            vector.set(i, 0, Math.exp(vector.get(i, 0)));
            sum += vector.get(i, 0);
        }
        output = vector.times(1/sum);
        return vector.times(1/sum);
    }

    public double cosine_sim(Matrix a, Matrix b)
    {
        double operand1 = 0;
        double operand2 = 0;
        double operand3 = 0;
        for (int i = 0; i < a.getRowDimension(); i++)
        {
            operand1 += a.get(i, 0) * a.get(i, 0);
            operand2 += b.get(i, 0) * b.get(i, 0);
            operand3 += a.get(i, 0) * b.get(i, 0);
        }
        return operand3 / (Math.sqrt(operand1) + Math.sqrt(operand2));
    }

    public double update_weights(String actual_word)
    {
        output = output_layer(context_words);
        error = output.minus(vectorize(actual_word));
        weight2 = weight2.minus((hidden.times(error.transpose())).times(learning_rate));
        /* N X V = N X V - ((N X 1) X T(V X 1))*/
        double scalar = 0;
        for (int i = 0; i < V; i++) {
            scalar += error.get(i, 0);
        }
        weight = weight.minus((weight2.transpose()).times(1 / (double) C).times(learning_rate).times(scalar));
        return cosine_sim(vectorize(actual_word), output);
    }

    public void train(int iter)
    {
        for (int t = 0; t < iter; t++ )
        {
            double sum = 0;
            for (int i = 0; i < training_set.size() - C; i++)
            {
                for (int j = 0; j < C + 1; j++)
                {
                    if (j != (C / 2) + 1) context_words.add(training_set.get(j));
                }
                sum += update_weights(training_set.get(i+2));
                context_words.clear();
            }
            System.out.println(String.format("%.20f", sum / (training_set.size() - C)));
        }
    }
}
