package xl.expr;

import java.io.IOException;
import java.util.Map;

import xl.util.XLCircularException;
import xl.util.XLException;

public interface XLModel {
    
    /**
     * Fetch the value of an expression in a cell with name name. 
     * @return String representation of the value in the cell. 
     */
    String getValue(String name); 

    /**
     * Returns the expression (to be shown in for instance the Editor)
     * @param name
     * @return String represenation of the expression in the cell
     */
    String getExpression(String name); 

    /**
     * Set the expression expr in cell name. 
     * @param name cell name 
     * @param expr expression to be input 
     * @throws IOException 
     * @throws XLException in case of syntax errors in input expression 
     * @throws XLCircularException in case of input expression with circular reference
     */
    void setCell(String name, String expr) throws IOException, XLException, XLCircularException;
}
