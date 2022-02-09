package xl.expr.cell;

import java.io.IOException;

import xl.expr.Environment;
import xl.util.XLCircularException;
import xl.util.XLException;

public class CircularTestCell extends ExprCell {

    private boolean alreadyVisited;


    public CircularTestCell  (String expr) throws IOException, XLException {
        super(expr); 
        alreadyVisited = false; 
    }

    @Override
    public double value(Environment env) throws XLCircularException {
        if (alreadyVisited) {
            throw new XLCircularException("circular entry"); 
        }

        alreadyVisited = true; 
        return super.value(env); 
    }

    @Override
    public String describe() {
        return "bomb"; // arbitrary
    }
    
}
