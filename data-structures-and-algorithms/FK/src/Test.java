import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;

public class Test {
	
	public static void permute(String s, PrintStream out) {
		permute(s, 0, out);
	}

	private static void permute(String s, int pos, PrintStream out) {
	    // basfall
	    if (pos == s.length() - 1) {
	        out.println(s.charAt(pos));
	    }
	    
	    for (int i = pos; i < s.length(); i++) {
	        StringBuilder sb = new StringBuilder(s); 
	        
	        // Byt plats på bokstav på plats i och pos
	        sb = switchPlace(sb, s, pos, i);
	                
	        permute(sb.toString(), pos + 1, out);
	        out.println(sb.charAt(pos));
	    }
	}
	
	private static StringBuilder switchPlace(StringBuilder sb, String s, int pos, int i) {
		char letterAtPos = s.charAt(pos);
        sb.setCharAt(pos, s.charAt(i));
        sb.setCharAt(i, letterAtPos);
        
        return sb; 
	}

	public static void main(String[] args) {
		String s = "ABC";
		StringBuilder sb = new StringBuilder(s);
		System.out.println(switchPlace(sb, s, 1, 2));
		
//		ByteArrayOutputStream baos = new ByteArrayOutputStream();
//		PrintStream ps = new PrintStream(baos);
//		permute("ABC", ps); 
//		String[] strArray = baos.toString().split("\\s+");
//		Arrays.sort(strArray);
//		for(String s : strArray) {
//		   System.out.println(s);
//		}
	}

}
