package xl.gui.menu;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JMenuItem;
import xl.gui.XL;

class ClearMenuItem extends JMenuItem implements ActionListener {

    private XL xl;

    public ClearMenuItem(XL xl) {
        super("Clear");
        this.xl = xl;
        addActionListener(this);
    }

    public void actionPerformed(ActionEvent e) {
        xl.clearSlot();
    }
}
