import java.io.*;


/**
 * @Description TODO
 * @Author kaideng
 **/
public class Test {


    public static void main(String[] args) {

        File file = new File("/data/projects/fate/fate-flow/logs/jobs/P0001E0001T001/fate_flow_run.log");


        try {
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file)));
            int x = 0;
            for (int i = 1; i <= 10000; i++) {
                try {
                    bw.write(Integer.toString(i));
                    bw.newLine();
                } catch (IOException e) {
                    e.printStackTrace();
                }

            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }


    }
}
