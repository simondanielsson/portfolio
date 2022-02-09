package mountain;

import java.util.HashMap;

import fractal.Fractal;
import fractal.TurtleGraphics; 

public class Mountain extends Fractal{
	private int length;
	private double dev;
	private HashMap<Side, Point> sideToMidPoint;
	
	public Mountain(int length, double dev) {
		this.length = length;
		this.dev = dev; 
		sideToMidPoint = new HashMap<Side, Point>();
	}

	@Override
	public String getTitle() {
		return "Mountain";
	}

	@Override
	public void draw(TurtleGraphics turtle) {
		// Setup first three points 
		Point p1 = new Point((int) Math.round(turtle.getWidth() / 2.0 - length / 2.0),
				(int) Math.round(turtle.getHeight() / 2 + Math.sqrt(3.0) * length / 4.0)); 
		Point p2 = new Point((int) Math.round(turtle.getWidth() / 2.0 + length / 2.0 + 50),
				(int) Math.round(turtle.getHeight() / 2 + Math.sqrt(3.0) * length / 4.0 + 40));
		Point p3 = new Point((int) Math.round(turtle.getWidth() / 2.0),
				(int) Math.round(turtle.getHeight() / 2 - Math.sqrt(3.0) * length / 4.0));
	
		// Populate side->midpoint map
		sideToMidPoint.put(new Side(p1, p2), midPoint(p1, p2, RandomUtilities.randFunc(dev)));
		sideToMidPoint.put(new Side(p2, p3), midPoint(p2, p3, RandomUtilities.randFunc(dev)));
		sideToMidPoint.put(new Side(p3, p1), midPoint(p3, p1, RandomUtilities.randFunc(dev)));
		
		// Initiate recursion
		fractalTriangle(turtle, order, p1, p2, p3, dev);		 // order from super class
	}
	
	private static Point midPoint(Point p1, Point p2, double rand) {
		return new Point((p1.getX() + p2.getX())/2, (int) Math.round((p1.getY() + p2.getY())/2 + rand));		
	}
	
	private Point getMidPoint(Point p1, Point p2, double dev) {
		Side side12 = new Side(p1, p2);
		Point midPoint = null;
		
		if (sideToMidPoint.containsKey(side12)) {
			midPoint = sideToMidPoint.remove(side12); 
		} else {
			midPoint = midPoint(p1, p2, RandomUtilities.randFunc(dev));
			sideToMidPoint.put(side12, midPoint);
		}
		
		return midPoint; 
	}
	
	public void fractalTriangle(TurtleGraphics turtle, int order, Point p1, Point p2, Point p3, double dev) {
		if (order == 0) {
			turtle.moveTo(p1.getX(), p1.getY());
			turtle.forwardTo(p2.getX(), p2.getY());
			turtle.forwardTo(p3.getX(), p3.getY());
			turtle.forwardTo(p1.getX(), p1.getY());
		} else {
//			// D1-D4
//			Point p12 = midPoint(p1, p2, RandomUtilities.randFunc(dev));
//			Point p23 = midPoint(p2, p3, RandomUtilities.randFunc(dev));
//			Point p31 = midPoint(p3, p1, RandomUtilities.randFunc(dev));	
			
			// D5
			Point p12 = getMidPoint(p1, p2, dev);
			Point p23 = getMidPoint(p2, p3, dev);
			Point p31 = getMidPoint(p3, p1, dev);
			
			// Construct 4 triangles out of first triangle
			fractalTriangle(turtle, order-1, p1, p12, p31, dev/2);
			fractalTriangle(turtle, order-1, p12, p23, p31, dev/2);
			fractalTriangle(turtle, order-1, p12, p2, p23, dev/2);
			fractalTriangle(turtle, order-1, p31, p23, p3, dev/2);
		}
		
	}
}
