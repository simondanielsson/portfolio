package xl.gui;

import java.awt.Color;
import javax.swing.JTextField;

public class Editor extends JTextField {

    private XL xl;

    public Editor(XL xl) {
        setBackground(Color.WHITE);
        addActionListener(e -> xl.getController().updateModel(e.getActionCommand()));
    }
}
