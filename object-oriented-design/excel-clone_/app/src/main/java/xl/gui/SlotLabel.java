package xl.gui;

import java.awt.Color;
import javax.swing.JButton;
import java.awt.event.MouseEvent;
import java.awt.event.MouseAdapter;

import xl.controller.Controller;


public class SlotLabel extends ColoredLabel{
    private XL xl;
    private String address;

    public SlotLabel(XL xl, String address) {
        super("                    ", Color.WHITE, RIGHT);
        this.xl = xl;
        this.address = address;
        addMouseListener(new SlotClicked());
    }

    public String getAddress() {
        return address;
    }

    class SlotClicked extends MouseAdapter {
        public void mouseClicked(MouseEvent e)  
        {  
            Controller controller = xl.getController();
            controller.setCurrent(address);  
        } 
    }
}
