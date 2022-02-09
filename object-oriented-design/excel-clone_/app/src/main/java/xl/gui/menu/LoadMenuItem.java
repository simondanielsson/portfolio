package xl.gui.menu;

import java.io.FileNotFoundException;
import javax.swing.JFileChooser;
import xl.gui.StatusLabel;
import xl.gui.XL;
import xl.gui.XLBufferedReader;
import java.util.Map;
import java.util.TreeMap;
import java.io.IOException;
import xl.util.*;

//added:
import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.util.Scanner; // Import the Scanner class to read text files

class LoadMenuItem extends OpenMenuItem {

    public LoadMenuItem(XL xl, StatusLabel statusLabel) {
        super(xl, statusLabel, "Load");
    }

    protected void action(String path) throws FileNotFoundException {

        Map<String,String> map = new TreeMap<String,String>();
        XLBufferedReader reader = new XLBufferedReader(path);
        reader.load(map);
        try {
            reader.close();
        }
        catch(IOException e) {
            e.printStackTrace();
        }
        super.xl.loadModel(map);
        System.out.println(map);
    }

    protected int openDialog(JFileChooser fileChooser) {
        return fileChooser.showOpenDialog(xl);
    }
}
