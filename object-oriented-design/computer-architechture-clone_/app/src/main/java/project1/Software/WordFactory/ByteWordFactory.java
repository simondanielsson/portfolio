package project1.Software.WordFactory;

import project1.Hardware.Operand.Word.ByteWord;
import project1.Hardware.Operand.Word.Word;
import project1.Hardware.Operand.Word.WordFactory.WordFactory;

public class ByteWordFactory implements WordFactory {

    public ByteWordFactory  () {
    }

    @Override
    public Word word(String s) {
        var value = Byte.parseByte(s); 
        return new ByteWord(value); 
    }
    
}
