package project1.Hardware.Operand.Adress;

import project1.Hardware.Memory.Memory;
import project1.Hardware.Operand.Operand;
import project1.Hardware.Operand.Word.Word;

public class Adress implements Operand {
    
    private int index;

    public Adress  (int index) {
        this.index = index; 
    }

    @Override
    public Word getWord(Memory memory) {
        return memory.wordAt(index); 
    }
    
    public String toString() {
        return "[" + String.valueOf(index) + "]";
    }
}
