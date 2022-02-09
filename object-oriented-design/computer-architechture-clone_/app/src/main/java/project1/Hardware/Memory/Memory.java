package project1.Hardware.Memory;

import java.util.ArrayList;
import java.util.List;

import project1.Hardware.Operand.Word.Word;
import project1.Hardware.Operand.Word.WordFactory.WordFactory;

public class Memory {

    private List<Word> words; 

    public Memory  (int nWords, WordFactory wf) {
        words = new ArrayList<>(nWords); 

        // Populate memory with empty words (of correct type)
        for (int i = 0; i < nWords; i++) {
            
            // The default value of the words has to be a integer-parseable String
            words.add(wf.word("0")); 
        }
    }
    
    /**
     * Returns word at the specified index in memory  
     * @param index: index of word to be fetched
     * @return word at index.
     */
    public Word wordAt(int index) {
        return words.get(index); 
    }
}
