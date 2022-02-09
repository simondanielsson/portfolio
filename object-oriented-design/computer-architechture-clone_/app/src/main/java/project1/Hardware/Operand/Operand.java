package project1.Hardware.Operand;

import project1.Hardware.Memory.Memory;
import project1.Hardware.Operand.Word.Word;

public interface Operand {

    /**
     * Returns the underlying Word object which is referenced to by this.
     * @param memory: The computer's memory
     * @return Word referenced to by object implementing this interface.
     */
    Word getWord(Memory memory); 
}