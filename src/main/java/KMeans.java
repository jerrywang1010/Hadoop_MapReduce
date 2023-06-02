import java.io.*;
import java.util.*;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMeans {
    private static int k;
    private static final int maxIter = 20;
    public static class Point {
        public Point(String s) {
            String[] arr = s.split(",");
            this.x = Double.parseDouble(arr[0]);
            this.y = Double.parseDouble(arr[1]);
        }

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }

        private final double x;
        private final double y;

        public double getY() {
            return y;
        }

        public double getX() {
            return x;
        }

        public Point add(Point p) {
            return new Point((this.x + p.x), (this.y + p.y));
        }

        public double distanceTo(Point p) {
            double dx = this.x - p.getX();
            double dy = this.y - p.getY();
            return Math.sqrt(dx * dx + dy * dy);
        }

        @Override
        public String toString() {
            return this.x + "," + this.y;
        }
    }

    public static void initializeCentroid(Configuration conf, Path pointsPath) throws IOException {
        Path centerPath = new Path("centroid.txt");

        FileSystem fs = FileSystem.get(conf);

        if (fs.exists(centerPath)) {
            fs.delete(centerPath, true);
        }

        List<Point> allPoints = new ArrayList<>();
        FSDataInputStream inputStream = fs.open(pointsPath);
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        String line;
        while ((line = reader.readLine()) != null) {
            Point p = new Point(line);
            allPoints.add(p);
        }

        List<Point> centroids = new ArrayList<>();
        // choose first k points as centroid
        for (int i = 0; i < KMeans.k; i ++) {
            centroids.add(allPoints.get(i));
        }
//        int sliceSize = allPoints.size() / KMeans.k;
//        for (int i = 0; i < KMeans.k - 1; i ++) {
//            Point sum = new Point(0, 0);
//            for (int j = i * sliceSize; j < (i + 1) * sliceSize; j ++) {
//                sum = sum.add(allPoints.get(j));
//            }
//            Point centroid = new Point(sum.getX() / sliceSize, sum.getY() / sliceSize);
//            centroids.add(centroid);
//        }
//
//        // last centroid
//        Point sum = new Point(0, 0);
//        for (int i = (KMeans.k - 1) * sliceSize; i < allPoints.size(); i ++) {
//            sum = sum.add(allPoints.get(i));
//        }
//        Point lastCentroid = new Point(sum.getX() / (allPoints.size() - (KMeans.k - 1) * sliceSize),
//                                sum.getY() / (allPoints.size() - (KMeans.k - 1) * sliceSize));
//        centroids.add(lastCentroid);

        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fs.create(centerPath, true)));
        for (Point p : centroids) {
            System.out.println("initial centroid = " + p.toString());
            bw.write(p.toString());
            bw.newLine();
        }
        bw.close();
    }

    public static class PointsMapper extends Mapper<LongWritable, Text, Text, Text> {

        List<Point> centroids = new ArrayList<>();
        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);
            Configuration conf = context.getConfiguration();

            // retrieve file path
            Path centroidsPath = new Path(conf.get("centroidPath"));

            // create a filesystem object
            FileSystem fs = FileSystem.get(conf);

            // create a file reader

            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(centroidsPath)));

            // read centroids from the file and store them in a centroids variable
            String line;
            while ((line = br.readLine()) != null) {
                centroids.add(new Point(line));
            }
            br.close();
        }

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            // input -> key: character offset, value -> a point (in Text)
            // write logic to assign a point to a centroid
            // emit key (centroid id/centroid) and value (point)
            double minDistance = Double.MAX_VALUE;
            int centroidIndex = 0;
            Point p = new Point(value.toString());
            for (int i = 0; i < centroids.size(); i++) {
                double distance = p.distanceTo(centroids.get(i));
                if (distance < minDistance) {
                    centroidIndex = i;
                    minDistance = distance;
                }
            }
            context.write(new Text(Integer.toString(centroidIndex)), new Text(p.toString()));
        }
    }


    public static class PointsReducer extends Reducer<Text, Text, Text, Text> {

        List<Point> newCentroids = new ArrayList<>();

        @Override
        public void setup(Context context) {

        }

        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            // Input: key -> centroid id/centroid , value -> list of points
            // calculate the new centroid
            // new_centroids.add() (store updated centroid in a variable)
            Point sum = new Point(0, 0);
            int count = 0;
            for (Text t : values) {
                sum = sum.add(new Point(t.toString()));
                count ++;
            }
            // Point centroid = new Point(sum.getX() / count, sum.getY() / count);
            Point centroid = new Point(sum.getX() / count, sum.getY() / count);
            newCentroids.add(centroid);
            context.write(key, new Text(centroid.toString()));
        }

        @Override
        public void cleanup(Context context) throws IOException, InterruptedException {
            // BufferedWriter
            // delete the old centroids
            // write the new centroids
            // super.setup(context);
            Configuration conf = context.getConfiguration();
            Path centroidPath = new Path(conf.get("centroidPath"));
            // create a filesystem object
            FileSystem fs = FileSystem.get(conf);
            if (fs.exists(centroidPath)) {
                fs.delete(centroidPath, true);
            }
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fs.create(centroidPath, true)));
            for (Point p : newCentroids) {
                bw.write(p.toString());
                bw.newLine();
            }
            bw.close();

            newCentroids.clear();
        }

    }

    public static int run1Iter(Configuration conf, String[] args) throws Exception {
        long startTime = System.currentTimeMillis();
        Job job = Job.getInstance(conf, "Kmean");
        job.setJarByClass(KMeans.class);

        job.setMapperClass(PointsMapper.class);
        job.setReducerClass(PointsReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));

        FileSystem fs = FileSystem.get(conf);
        Path outputPath = new Path(args[1]);
        if (fs.exists(outputPath)) {
            fs.delete(outputPath, true);
        }
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        boolean success = job.waitForCompletion(true);
        long duration = System.currentTimeMillis() - startTime;
        System.out.println("Mapreduce execution time=" + duration + "ms");
        return success ? 0 : -1;
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.out.println("Usage: Kmean3.jar $inputPath $outputPath $k");
            System.exit(1);
        }
        KMeans.k = Integer.parseInt(args[2]);
        Configuration conf = new Configuration();

        Path centroidPath = new Path("centroid.txt");
        conf.set("centroidPath", centroidPath.toString());
        initializeCentroid(conf, new Path(args[0]));
        int iter = 0;
        long startTime = System.currentTimeMillis();
        while (iter < KMeans.maxIter) {
            System.out.println("Iteration=" + iter);
            int exitCode = run1Iter(conf, args);
            if (exitCode != 0) {
                System.out.println("run1Iter returned non-zero exit code, exiting");
                return;
            }
            iter ++;
        }
        long duration = System.currentTimeMillis() - startTime;
        System.out.println("Total execution time=" + duration + "ms");
    }

}
