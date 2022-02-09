package xl.expr;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import xl.expr.cell.*;
import xl.util.CellNameGenerator;
import xl.util.XLCircularException;
import xl.util.XLException;

public class XLTreeModel implements XLModel, Environment {
    
    // Names of cells in the model (migh be empty cells)
    private List<String> cellNames; 

    // Occupied cells
    private Map<String, Cell> cells = new TreeMap<>();

    public XLTreeModel  (int rows, int cols) {
        cellNames = new CellNameGenerator(rows, cols).generateNames(); 
    }

    /**
     * Safe cell-getter
     */ 
    private Cell getCellSafely(String name) {
        // Trying to fetch value outside the spreadsheet
        if (!cellNames.contains(name)) throw new IllegalArgumentException("key " + name + " not found"); 

        // Reference to an empty cell
        if (!cells.containsKey(name)) {
            return EmptyCell.getCell(); 
        }

        return cells.get(name); 
    }

    /**
     * To be used internally by the cells when calculating their values. 
     */
    @Override
    public double value(String name) {
        return getCellSafely(name).value(this); 
    }

    @Override 
    public String getValue(String name) {
        var cell = getCellSafely(name);

        if (cell instanceof CommentCell) {
            return cell.describe().substring(1); 
        } 

        if (cell instanceof EmptyCell) {
            return cell.describe(); 
        }

        return String.valueOf(
            cell.value(this)
        );
    }

    @Override
    public String getExpression(String name) {
        return getCellSafely(name).describe();
    }

    @Override
    public void setCell(String name, String expression) 
        throws IOException, XLException, XLCircularException {
        
        if (expression.isEmpty()) {
            cells.remove(name); 
            return;
        }
        
        else if (isComment(expression)) { 
            cells.put(name, new CommentCell(expression)); 
            return;
        } 

        checkForCircularity(name, expression);
        putExpressionCell(name, expression);
        
    }

    private boolean isComment(String expression) {
        return expression.charAt(0) == '#'; 
    }

    /**
     * Check for circularity in the input expression: if circularity 
     * is reached, put back the previous cell
     */
    private void checkForCircularity(String name, String expression) 
        throws IOException, XLException, XLCircularException {

        // Check for circularity by injecting test cell into the system
        var previousCell = getCellSafely(name);

        // Try to create the circular test cell
        CircularTestCell testCell;
        try {
            testCell = new CircularTestCell(expression);
        } catch (IOException e) {
            throw new IOException("io exception: something went wrong");
        } catch (XLException e) {
            throw new XLException("syntax error: could not construct expression"); 
        }
        cells.put(name, testCell);
        
        // Try to evaluate its value: if cicularity is reached, 
        // XLCircularException will be thrown by the cell
        try {
            testCell.value(this); 
        } catch (XLException e) {
            cells.put(name, previousCell); 
            throw new XLException("syntax error: could not compute value");
        } catch (XLCircularException e) {
            cells.put(name, previousCell); 
            throw new XLCircularException("circular statement when computing value"); 
        }
    } 

    /**
     * Add expression cell to the model.
     */
    private void putExpressionCell(String name, String expression) 
        throws IOException, XLException {
        
        // Create actual expression cell.
        Cell exprCell;
        try {
            exprCell = new ExprCell(expression); 
        } catch (IOException e) {
            throw new IOException("io exception");
        } catch (XLException e) {
            throw new XLException("syntax error"); 
        }

        cells.put(name, exprCell); 
    }
}
