import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount_NoCombiner {
    // letters 'm', 'n', 'o', 'p',and 'q'
    private static final Map<Character, Integer> letters = new HashMap<Character, Integer>() {{
        put('m', 0);
        put('n', 1);
        put('o', 2);
        put('p', 3);
        put('q', 4);
    }};

    public static class WordPartitioner extends Partitioner<Text, IntWritable> {
        @Override
        public int getPartition(Text key, IntWritable value, int numPartitions) {
            char firstChar = key.toString().toLowerCase().charAt(0);
            return letters.get(firstChar);
        }
    }

    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable>{
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String token = itr.nextToken();
                word.set(token);

                // Step 1 - Look to refactor later
                if (!token.isEmpty()) {
                    char firstChar = Character.toLowerCase(token.charAt(0));
                    if (letters.containsKey(firstChar)) {
                        context.write(word, one);
                    }
                }
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount_NoCombiner.class);
        job.setMapperClass(TokenizerMapper.class);
        // job.setCombinerClass(IntSumReducer.class); // Combiner is disabled
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setPartitionerClass(WordPartitioner.class);
        job.setNumReduceTasks(letters.size());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}