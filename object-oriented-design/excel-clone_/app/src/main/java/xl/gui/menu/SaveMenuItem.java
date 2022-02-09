package xl.gui.menu;

import java.io.FileNotFoundException;
import javax.swing.JFileChooser;
import xl.gui.StatusLabel;
import xl.gui.XL;

// added:
import java.io.FileWriter; 
import java.io.IOException;

class SaveMenuItem extends OpenMenuItem {

    public SaveMenuItem(XL xl, StatusLabel statusLabel) {
        super(xl, statusLabel, "Save");
    }

    protected void action(String path) throws FileNotFoundException {

        StringBuilder builder = new StringBuilder(); 
        for (var entry : xl.getSlotLabels()) {
            var expr = this.xl.getModel().getExpression(entry.getAddress());
            if(!expr.isEmpty()) {
                builder.append(entry.getAddress());
                builder.append("=");
                builder.append(expr);
                builder.append("\n");
            }
        }
        String str = builder.toString();
        try {
            FileWriter writer = new FileWriter(path);
            writer.write(str);
            writer.close();
          } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
          }
    }

    protected int openDialog(JFileChooser fileChooser) {
        return fileChooser.showSaveDialog(xl);
    }
}
