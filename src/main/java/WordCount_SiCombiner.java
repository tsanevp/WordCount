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

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

/**
 * A Hadoop MapReduce program that counts words starting with
 * the letters 'm', 'n', 'o', 'p', 'q', using a combiner to
 * aggregate the intermediate results at the mapper level.
 */
public class WordCount_SiCombiner {

    /**
     * Map of characters 'm', 'n', 'o', 'p', and 'q'.
     * Used to partition words into different reducers and
     * to filter words based on their first letter.
     */
    private static final Map<Character, Integer> letters = new HashMap<Character, Integer>() {{
        put('m', 0);
        put('n', 1);
        put('o', 2);
        put('p', 3);
        put('q', 4);
    }};

    /**
     * The main method that sets up the Hadoop MapReduce job configuration.
     * It specifies the Mapper, Reducer, Combiner, Partitioner, and other job parameters.
     *
     * @param args Command line arguments: input and output file paths.
     * @throws Exception If an error occurs during job configuration or execution.
     */
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(WordCount_SiCombiner.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(IntSumReducer.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setPartitionerClass(WordPartitioner.class);
        job.setNumReduceTasks(letters.size());

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    /**
     * A custom Partitioner class that partitions words by their
     * first character. Words starting with 'm' go to reducer 0,
     * 'n' to reducer 1, and so on.
     */
    public static class WordPartitioner extends Partitioner<Text, IntWritable> {
        /**
         * Assigns a partition to each word based on its first character.
         *
         * @param key           The word to partition.
         * @param value         The count associated with the word.
         * @param numPartitions The total number of partitions (reducers).
         * @return The partition number for the word.
         */
        @Override
        public int getPartition(Text key, IntWritable value, int numPartitions) {
            char firstChar = key.toString().toLowerCase().charAt(0);
            return letters.get(firstChar);
        }
    }

    /**
     * The Mapper class for tokenizing lines of input into words,
     * emitting each word with a count of 1 if it starts with one
     * of the target letters ('m', 'n', 'o', 'p', 'q').
     */
    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);
        private final Text word = new Text();

        /**
         * The map method processes each line of input, tokenizes it,
         * and emits each word with a count of 1 if the word starts
         * with the letters 'm', 'n', 'o', 'p', or 'q'.
         *
         * @param key     The input key (usually the byte offset).
         * @param value   The input value (a line of text).
         * @param context The context for writing output key-value pairs.
         * @throws IOException          If an I/O error occurs.
         * @throws InterruptedException If the map task is interrupted.
         */
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer itr = new StringTokenizer(value.toString());
            while (itr.hasMoreTokens()) {
                String token = itr.nextToken();
                word.set(token);

                // Check if the token starts with one of the target letters
                if (!token.isEmpty()) {
                    char firstChar = Character.toLowerCase(token.charAt(0));
                    if (letters.containsKey(firstChar)) {
                        context.write(word, one);
                    }
                }
            }
        }
    }

    /**
     * The Reducer class for summing the counts of words from the map tasks.
     * It receives the word and its count from each mapper and aggregates
     * the count across all mappers.
     */
    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private final IntWritable result = new IntWritable();

        /**
         * The reduce method aggregates the word counts by summing
         * the counts for each word.
         *
         * @param key     The word being reduced.
         * @param values  The counts of the word from each map task.
         * @param context The context for writing the final output.
         * @throws IOException          If an I/O error occurs.
         * @throws InterruptedException If the reduce task is interrupted.
         */
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }
}