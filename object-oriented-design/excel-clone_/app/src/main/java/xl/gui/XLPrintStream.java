package xl.gui;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Map.Entry;
import java.util.Set;

public class XLPrintStream extends PrintStream {

    public XLPrintStream(String fileName) throws FileNotFoundException {
        super(fileName);
    }

    public void save(Set<Entry<String, String>> set) {
        for (Entry<String, String> entry : set) {
            print(entry.getKey());
            print('=');
            println(entry.getValue());
        }
        flush();
        close();
    }
}
