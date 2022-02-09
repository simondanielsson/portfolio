package xl.gui.menu;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JMenuItem;
import xl.gui.XL;

class ClearAllMenuItem extends JMenuItem implements ActionListener {

    private XL xl;

    public ClearAllMenuItem(XL xl) {
        super("Clear all");
        this.xl = xl;
        addActionListener(this);
    }

    public void actionPerformed(ActionEvent e) {
        xl.clearModel();
    }
}
