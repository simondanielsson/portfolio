package textproc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class GeneralWordCounter implements TextProcessor {
	Map<String, Integer> words;
	Set<String> stopwords;
	
	public GeneralWordCounter(Set<String> stopwords) {
		words = new HashMap<String, Integer>();
		this.stopwords = stopwords;
	}
	
	public void process(String word) {
		if (!stopwords.contains(word)) {  
			if (!words.containsKey(word)) {
				words.put(word, 0);
			} 
			words.put(word, words.get(word) + 1);
		}
	}
	
	public void report() {
//		StringBuilder sb = new StringBuilder(); 
//		
//		for (String word : words.keySet()) {
//			int frequency = words.get(word);
//			if (frequency >= 200) {
//				sb.append(word + ": " + frequency + "\n"); 
//			}
//		}
//		
//		System.out.println(sb);
		
		Set<Map.Entry<String, Integer>> wordSet = words.entrySet();
		List<Map.Entry<String, Integer>> wordList = new ArrayList<>(wordSet);
		
		wordList.sort(new WordCountComparator());
		
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < 15; i++) {
			sb.append(wordList.get(i) + "\n");
		}
		System.out.println(sb);
		
		
	}

}
