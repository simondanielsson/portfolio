package project1.Hardware.Operand.Word.WordFactory;

import project1.Hardware.Operand.Word.Word;

public interface WordFactory {

    /**
     * Factory object returning a specific word instance, containing 
     * information from input string s. 
     * @return instance of some class generalizing Word. 
     */
    public Word word(String s); 
}
