package project1.Hardware.Operand.Word;

import project1.Hardware.Memory.Memory;

public class ByteWord extends Word {

    private byte value; 

    public ByteWord  (byte value) {
        this.value = value;
    }

    @Override
    public Word getWord(Memory memory) {
        return this;
    }

    private ByteWord ref(Word word) {
        return (ByteWord) word; 
    }

    @Override
    public void add(Word w1, Word w2) {
        value = (byte) (ref(w1).value + ref(w2).value);  //varför behöver vi byte-type casta här
    }

    @Override
    public void mul(Word w1, Word w2) {
        value = (byte) (ref(w1).value * ref(w2).value); 
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