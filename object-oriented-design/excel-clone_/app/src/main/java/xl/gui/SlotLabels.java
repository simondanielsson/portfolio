package xl.gui;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;
import javax.swing.SwingConstants;


public class SlotLabels extends GridPanel {

    private List<SlotLabel> labelList;
    private XL xl;

    public SlotLabels(int rows, int cols, XL xl) {
        super(rows + 1, cols);
        this.xl = xl;
        labelList = new ArrayList<SlotLabel>(rows * cols);
        for (char ch = 'A'; ch < 'A' + cols; ch++) {
            add(new ColoredLabel(Character.toString(ch), Color.LIGHT_GRAY, SwingConstants.CENTER));
        }
        for (int row = 1; row <= rows; row++) {
            for (char ch = 'A'; ch < 'A' + cols; ch++) {
                SlotLabel label = new SlotLabel(xl, ch + Integer.toString(row)); 
                add(label);
                labelList.add(label);
            }
        }
        SlotLabel firstLabel = labelList.get(0);
        firstLabel.setBackground(Color.YELLOW);
    }

    public SlotLabel getLabel(String address) {
        for(var slot: labelList) {
            if(slot.getAddress().equals(address)) {
                return slot;
            }
        }
        return null;
    }

    public List<SlotLabel> getLables(){
        return labelList;
    }
}
