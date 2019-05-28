package controllers

import javax.inject._

import play.api.mvc._
import com.scalaml.nn.Playground

/**
 * This controller demonstrates how to use dependency injection to
 * bind a component into a controller class. The class creates an
 * `Action` that shows an incrementing count to users. The [[Counter]]
 * object is injected by the Guice dependency injection system.
 */
@Singleton
class PgController @Inject() (cc: ControllerComponents,
                                 pg: Playground) extends AbstractController(cc) {

    /**
     * Create an action that responds with the [[Counter]]'s current
     * count. The result is plain text. This `Action` is mapped to
     * `GET /count` requests by an entry in the `routes` config file.
     */
    def test = Action {
        pg.generateData()
        pg.reset()
        for(_ <- 1 to 1000) pg.oneStep()
        Ok(pg.report.toString)
    }
    def init = Action {
        pg.generateData()
        pg.reset()
        Ok(pg.report.toString)
    }
    def gen = Action {
        pg.generateData()
        Ok(pg.report.toString)
    }
    def reset = Action {
        pg.reset()
        Ok(pg.report.toString)
    }
    def one = Action {
        pg.oneStep()
        Ok(pg.report.toString)
    }
    def run(_iter: String) = Action {
        for(_ <- 1 to _iter.toInt) pg.oneStep()
        Ok(pg.report.toString)
    }
    def state = Action {
        Ok(pg.state.toString)
    }
    def report = Action {
        Ok(pg.report.toString)
    }

}
