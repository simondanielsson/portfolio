package project1.Hardware.Operand.Word;

import project1.Hardware.Memory.Memory;
import project1.Hardware.Operand.Operand;

public abstract class Word implements Operand {

    public Word getWord(Memory memory) { 
        return this;
    }
    
    /**
     * Add words w1 and w2, and store the sum in this. 
     * @param w1: word to add
     * @param w2: another word to add
     */
    public abstract void add(Word w1, Word w2); 

    /**
     * Multiply words w1 and w2, and store the product in this. 
     * @param w1: word to add
     * @param w2: another word to add
     */
    public abstract void mul(Word w1, Word w2); 

    /**
     * Checks if other word is equal to this
     * @param other: another word
     * @return true if this and other should be considered equal, else false
     */
    public abstract boolean equals(Word other); 

    /**
     * Returns a String-representation of this word
     */
    public abstract String toString(); 

    /**
     * Copy the contents of other into this.
     * @param other: Word to copy information from
     */
    public abstract void copy(Word other);
}
