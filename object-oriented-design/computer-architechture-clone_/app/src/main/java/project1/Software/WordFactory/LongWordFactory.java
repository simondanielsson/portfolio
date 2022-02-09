package project1.Software.WordFactory;

import project1.Hardware.Operand.Word.LongWord;
import project1.Hardware.Operand.Word.Word;
import project1.Hardware.Operand.Word.WordFactory.WordFactory;

public class LongWordFactory implements WordFactory {

    public LongWordFactory  () {
    }

    @Override
    public Word word(String s) {
        var value = Long.parseLong(s);
        return new LongWord(value); 
    }
    
}
