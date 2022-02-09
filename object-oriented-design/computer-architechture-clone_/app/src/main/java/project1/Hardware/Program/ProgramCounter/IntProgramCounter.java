package project1.Hardware.Program.ProgramCounter;

public class IntProgramCounter implements ProgramCounter {

    private int index; 
    private boolean stillRunning;

    public IntProgramCounter  () {
        index = 0; 
        stillRunning = true; 
    }

    @Override
    public int getIndex() {
        return index;
    }

    @Override
    public void jumpTo(int newIndex) {
        index = newIndex; 
    }
    
    @Override
    public void step() {
        index++; 
    }

    @Override
    public void halt() {    
        stillRunning = false; 
    }

    @Override
    public boolean stillRunning() {
        return stillRunning; 
    }
}