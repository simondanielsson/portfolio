
package xl.controller;

import xl.expr.XLModel;
import xl.gui.XLView;
import xl.util.*;
import java.io.IOException;

public class XLController implements Controller {

    private XLModel model;
    private XLView view;
    private String currentCell;

    public XLController (XLModel model, XLView view, String currentCell){
        this.model = model;
        this.view = view;
        this.currentCell = currentCell;
    }

    public void setCurrent(String newCurrent){
        String oldCurrent = currentCell;
        currentCell = newCurrent;
        view.updateCurrent(oldCurrent, newCurrent);
        view.reportError("");
    }

    public String getCurrent(){
        return currentCell;
    }
    

    public void updateModel(String input){
        try{
            String cell = getCurrent();
            model.setCell(cell, input);
            view.reportError("");
        }
        catch (IOException e){
            //update status panel "io error"
            view.reportError("IO error");
            return;
        }
        catch (XLException e){
            //update status panel "syntax error"
            view.reportError("Syntax error: illegal input");
            return;
        }
        catch (XLCircularException e) {
            //update status panel "circular expression"
            view.reportError("Circular expression");
            return;
        }

        view.updateView();

    } 

    

}
