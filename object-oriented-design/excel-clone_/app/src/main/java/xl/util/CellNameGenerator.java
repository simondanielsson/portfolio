package xl.util;

import java.util.ArrayList;
import java.util.List;

/**
 * Class for generating cell names. 
 */
public class CellNameGenerator {

    private int rows; 
    private int cols; 

    public CellNameGenerator  (int rows, int cols) {
        this.rows = rows; 
        this.cols = cols;
    }

    public List<String> generateNames() {
        List<String> names = new ArrayList(rows * cols);

        for (var letter : letters()) {
            for (var number : numbers()) {
                names.add(letter + number); 
            }
        }

        return names; 
    }

    private List<String> letters() {
        if (cols >= 26) {
            throw new IllegalArgumentException(
                "have to redo the way the column names are initialized in the model, not enough letters A-Z"
            );
        }

        var letters = new ArrayList<String>(); 

        for (int i = 0; i < cols; i++) {
            letters.add(String.valueOf((char) ('A' + i))); 
        }
        
        return letters; 
    }

    private List<String> numbers() {
        var numbers = new ArrayList<String>();
        
        for (int i = 1; i <= rows; i++) {
            numbers.add(String.valueOf(i)); 
        }
        
        return numbers;
    }
}
