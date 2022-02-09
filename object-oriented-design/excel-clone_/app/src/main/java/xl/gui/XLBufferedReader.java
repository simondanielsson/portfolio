package xl.gui;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Map;
import xl.util.XLException;


public class XLBufferedReader extends BufferedReader {

    public XLBufferedReader(String name) throws FileNotFoundException {
        super(new FileReader(name));
    }

    public void load(Map<String, String> map) {
        try {
            while (ready()) {
                String string = readLine();
                int i = string.indexOf('=');
                map.put(string.substring(0, i), string.substring(i+1));
            }
        } catch (Exception e) {
            throw new XLException(e.getMessage());
        }
    }
}
