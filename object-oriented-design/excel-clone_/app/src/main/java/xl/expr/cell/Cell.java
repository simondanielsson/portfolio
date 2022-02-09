package xl.expr.cell;

import xl.expr.Environment;

public interface Cell {

    double value(Environment env); 
    String describe(); 
}
