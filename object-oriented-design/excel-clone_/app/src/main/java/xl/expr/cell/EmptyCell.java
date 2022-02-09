package xl.expr.cell;

import xl.expr.Environment;


/**
 * Singleton representation of an empty cell
 */
public class EmptyCell implements Cell {

    private static EmptyCell instance = new EmptyCell(); 

    private EmptyCell  () {
    }

    public static EmptyCell getCell() {
        return instance; 
    }

    @Override
    public double value(Environment env) {
        return 0;
    }

    @Override
    public String describe() {
        return ""; 
    }
    
}
