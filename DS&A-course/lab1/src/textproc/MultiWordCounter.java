package textproc;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

public class MultiWordCounter implements TextProcessor {
	Map<String, Integer> words; 
	
	public MultiWordCounter(String[] landskap) {
		words = new TreeMap<String, Integer>(); 
		
		for (String s : landskap) {
			words.put(s, 0); 
		}
	}
	
	public void process(String word) {
		if (words.containsKey(word)) {
			words.put(word, words.get(word) + 1);
		}
	}
	
	public void report() {
		StringBuilder sb = new StringBuilder();
		
		for (String key : words.keySet()) {
			sb.append(key + ": " + words.get(key) + "\n");
		}
		
		System.out.println(sb);
	}
}
