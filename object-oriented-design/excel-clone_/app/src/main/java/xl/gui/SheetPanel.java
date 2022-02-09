package xl.gui;

import static java.awt.BorderLayout.CENTER;
import static java.awt.BorderLayout.WEST;

public class SheetPanel extends BorderPanel {

    private SlotLabels slotLabels;

    public SheetPanel(int rows, int columns, XL xl) {
        add(WEST, new RowLabels(rows));
        this.slotLabels = new SlotLabels(rows, columns, xl);
        add(CENTER, slotLabels);
    }


    public SlotLabels getSlotLabels() {
        return slotLabels;
    }

}
