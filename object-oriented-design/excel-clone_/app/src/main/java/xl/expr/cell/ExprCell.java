package xl.expr.cell;

import java.io.IOException;

import xl.expr.Environment;
import xl.expr.Expr;
import xl.expr.ExprParser;
import xl.util.XLException;

public class ExprCell implements Cell {
    
    private Expr expr;

    public ExprCell  (String expression) throws XLException, IOException {
        var parser = new ExprParser(); 

        try {
            expr = parser.build(expression); 
        } catch (IOException e) {
            throw new IOException("io error"); 
        } catch (XLException e) {
            throw new XLException("syntax error"); 
        }     
    }

    @Override
    public double value(Environment env) {
        return expr.value(env); 
    }

    @Override
    public String describe() {
        return expr.toString(); 
    }
}