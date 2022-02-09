package textproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

public class Holgersson {

	public static final String[] REGIONS = { "blekinge", "bohuslän", "dalarna", "dalsland", "gotland", "gästrikland",
			"halland", "hälsingland", "härjedalen", "jämtland", "lappland", "medelpad", "närke", "skåne", "småland",
			"södermanland", "uppland", "värmland", "västerbotten", "västergötland", "västmanland", "ångermanland",
			"öland", "östergötland" };

	public static void main(String[] args) throws FileNotFoundException {
		long startTime = System.nanoTime();
		
//		List<TextProcessor> processors = new ArrayList<TextProcessor>();
//		processors.add(new SingleWordCounter("nils")); 
//		processors.add(new SingleWordCounter("norge")); 
		
//		TextProcessor multiCounter = new MultiWordCounter(REGIONS);
		
		Scanner scan = new Scanner(new File("undantagsord.txt"));
		Set<String> stopwords = new HashSet<String>();
	
		while (scan.hasNext()) {
			stopwords.add(scan.next());
		}
		
		TextProcessor generalCounter = new GeneralWordCounter(stopwords);
		
		Scanner s = new Scanner(new File("nilsholg.txt"));
		s.findWithinHorizon("\uFEFF", 1);
		s.useDelimiter("(\\s|,|\\.|:|;|!|\\?|'|\\\")+"); // se handledning
		
		while (s.hasNext()) {
			String word = s.next().toLowerCase();
			
//			for (TextProcessor p : processors) {
//				p.process(word);
//			}
			
//			multiCounter.process(word); 
			
			generalCounter.process(word);
		}

		s.close();
		
//		for (TextProcessor p : processors) {
//			p.report();
//		}
		
//		multiCounter.report(); 
		
		generalCounter.report();
		scan.close();
//		
		long endTime = System.nanoTime(); 	
		System.out.println("tid: " + (endTime - startTime) / 1000000.0 + " ms");

	}
}