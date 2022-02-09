package project1.Hardware.Program.ProgramCounter;

/**
 * Interface for keeping track of what index in the list of instructions to be run 
 * next in by program. 
 */
public interface ProgramCounter {    
    /**
     * Return index of this ProgramCounter
     * @return the index
     */
    int getIndex(); 

    /**
     * Set the index contained in this ProgramCounter to newIndex
     * @param newIndex: index to passed to the counter
     */
    void jumpTo(int newIndex); 

    void step(); 

    void halt(); 

    boolean stillRunning(); 
}
