package xl.expr.cell;

import xl.expr.Environment;

public class CommentCell implements Cell {

    // Comment with leading '#'. 
    private String comment;

    public CommentCell  (String comment) {
        this.comment = comment;
    }

    @Override
    public double value(Environment env) {
        return 0;
    }

    @Override
    public String describe() {
        return comment; 
    }
    
}
