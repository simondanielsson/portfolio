package xl.gui;

import static java.awt.BorderLayout.CENTER;
import static java.awt.BorderLayout.NORTH;
import static java.awt.BorderLayout.SOUTH;

import java.awt.Color;

import javax.swing.JFrame;
import javax.swing.JPanel;
import xl.gui.menu.XLMenuBar;

// added:
import xl.expr.XLModel;
import xl.expr.XLTreeModel;
import xl.gui.XLView;
import xl.controller.Controller;
import xl.controller.XLController;
import xl.gui.SheetPanel;
import xl.gui.SlotLabels;
import xl.gui.SlotLabel;
import java.util.Map;
import java.util.TreeMap;
import java.io.IOException;
import xl.util.XLException;
import xl.util.XLCircularException;
import java.util.List;

public class XL extends JFrame implements XLView{

    private static final int ROWS = 10, COLUMNS = 8;
    private XLCounter counter;
    private StatusLabel statusLabel = new StatusLabel();
    private XLList xlList;
    private XLModel model; 
    private StatusPanel statusPanel;
    private Controller controller;
    private SlotLabels slotLabels;
    private CurrentLabel currentLabel;
    private Editor editor;

    public XL(XL oldXL) {
        this(oldXL.xlList, oldXL.counter);
    }

    public XL(XLList xlList, XLCounter counter) {
        super("Untitled-" + counter);
        this.xlList = xlList;
        this.counter = counter;
        xlList.add(this);
        counter.increment();
        StatusPanel statusPanel = new StatusPanel(statusLabel);
        currentLabel = statusPanel.getCurrentLabel();
        SheetPanel sheetPanel = new SheetPanel(ROWS, COLUMNS, this);
        slotLabels = sheetPanel.getSlotLabels();
        editor = new Editor(this);
        add(NORTH, statusPanel);
        add(CENTER, editor);
        add(SOUTH, sheetPanel);
        setJMenuBar(new XLMenuBar(this, xlList, statusLabel));
        pack();
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setResizable(false);
        setVisible(true);

        model = new XLTreeModel(ROWS, COLUMNS); 
        controller = new XLController(model, this, "A1");
    }

    public XL(XLList xlList, XLCounter counter, XLModel model) {
        this(xlList, counter);
        this.model = model;
    }

    public void rename(String title) {
        setTitle(title);
        xlList.setChanged();
    }


    public XLModel getModel() {
        return this.model;
    }

    public Controller getController() {
        return this.controller;
    }

    public List<SlotLabel> getSlotLabels(){
        return slotLabels.getLables();
    }

    public static void main(String[] args) {
        new XL(new XLList(), new XLCounter());
    }

    public void updateView() {

        for (SlotLabel s : slotLabels.getLables()) {
            String adress = s.getAddress();
            String value = model.getValue(adress);
            s.setText(value);          
        }
    }

    public void reportError(String errorMessage) {
        statusLabel.update(errorMessage);
    }

    public void updateCurrent(String oldAddress, String newAddress) {
        SlotLabel oldCurrent = slotLabels.getLabel(oldAddress);
        oldCurrent.setBackground(Color.WHITE);
        SlotLabel newCurrent = slotLabels.getLabel(newAddress);
        newCurrent.setBackground(Color.YELLOW);
        currentLabel.setText(newAddress);
        String expr = model.getExpression(newAddress);
        editor.setText(expr);
    }

    public void loadModel(Map<String,String> map) {
        try {
            for(var entry: map.entrySet()) {
                String key = entry.getKey();
                String expr = entry.getValue();
                model.setCell(key, expr);
            }
            updateView();
        }
        catch (IOException e){
            //uppdatera status-rutan "io error"
            reportError("IO error");
            return;
        }
        catch (XLException e){
            //uppdatera status-rutan "syntax error"
            reportError("syntax error");
            return;
        }
        catch (XLCircularException e) {
            //uppdatera status-rutan "circular expression"
            reportError("circular expression");
            return;
        }
    }

    public void clearModel() {
        Map<String,String> zeroMap = new TreeMap<String,String>();
        for (int row = 1; row <= ROWS; row++) {
            for (char ch = 'A'; ch < 'A' + COLUMNS; ch++) {
                zeroMap.put(ch+Integer.toString(row), "");
            }
        }
        loadModel(zeroMap);
    }

    public void clearSlot() {
        controller.updateModel("");
    }
}
