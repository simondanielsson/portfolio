package project1.Hardware.Operand.Word;

import project1.Hardware.Memory.Memory;

public class LongWord extends Word {

    private long value;

    public LongWord  (long value) {
        this.value = value; 
    }

    @Override
    public Word getWord(Memory memory) {
        return this;
    }

    private LongWord ref(Word word) {
        return (LongWord) word; 
    }

    @Override
    public void add(Word w1, Word w2) {
        value = ref(w1).value + ref(w2).value;
    }

    @Override
    public void mul(Word w1, Word w2) {
        value = ref(w1).value * ref(w2).value;
    }

    @Override
    public boolean equals(Word other) {
        return value == ref(other).value; 
    }

    @Override
    public String toString() {
        return String.valueOf(value);
    }

    @Override
    public void copy(Word other) {
        value = ref(other).value; 
    }
    
}
