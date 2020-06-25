package testeweka;


import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class main {
    public static void main(String[] args) throws Exception {

        DataSource da = new DataSource("src/testeweka/vendas.arff");
        Instances ins = da.getDataSet();
        //System.out.println(ins.toString());

        ins.setClassIndex(3);

        NaiveBayes nb =  new NaiveBayes();
        nb.buildClassifier(ins);

        Instance novo = new DenseInstance(4);
        novo.setDataset(ins);
        novo.setValue(0,"M");
        novo.setValue(1,"20-39");
        novo.setValue(2,"Sim");

        double probabilidade[] = nb.distributionForInstance(novo);
        System.out.println("Sim: "+probabilidade[1]);
        System.out.println("Nao: "+probabilidade[0]);



    }
}
